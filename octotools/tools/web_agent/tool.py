import os
from octotools.tools.base import BaseTool
from octotools.engine.factory import create_llm_engine
from transformers import AutoTokenizer
import requests
from typing import List, Dict, Any
import re

# PREFIX = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question:"""

# TODO: summary mechanism needed
REASONING_SUMMARY_PREFIX = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
Once you have enough information, you must include the final answer together with a concise explanation of why this answer is correct, both inside the <answer> and </answer> tags. The explanation should summarize the key reasoning steps that led to the answer. For example: <answer>
The emperor of Japan during World War I was Emperor Taishō, who reigned from 1912 to 1926. Historical records show that his biological mother was Yanagihara Naruko, a concubine of Emperor Meiji. Therefore, based on this lineage, the mother of the emperor during World War I was Yanagihara Naruko.
</answer> Question:"""

class Google_Search():
    def __init__(self):
        # self.api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = os.getenv("GOOGLE_API_KEY") # NOTE: Replace with your own API key (Ref: https://developers.google.com/custom-search/v1/introduction)
        self.cx = os.getenv("GOOGLE_CX") # NOTE: Replace with your own custom search (Ref: https://programmablesearchengine.google.com/controlpanel/all)
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Performs a Google search using the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return.

        Returns:
            Dict[str, Any]: The raw search results from the Google API.
        """
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cx,
            'num': num_results
        }
        
        response = requests.get(self.base_url, params=params)
        return response.json()

    def execute(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Executes a Google search based on the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return (default: 10).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search result information.
        """
        if not self.api_key:
            return [{"error": "Google API key is not set. Please set the GOOGLE_API_KEY environment variable."}]

        try:
            results = self.google_search(query, num_results)
            
            if 'items' in results:
                return [
                    {
                        "title": item['title'],
                        "link": item['link'],
                        "snippet": item['snippet']
                    }
                    for item in results['items']
                ]
            else:
                return [{"error": "No results found."}]
        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]

class Web_Agent_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string): 
        super().__init__(
            tool_name="Web_Agent_Tool",
            tool_description=(
        "A web agent tool that integrates google search tool with reasoning ability "
        "to interpret user tasks, plan appropriate actions, autonomously invoke tool, and generate intelligent responses "
        "based on online information."
    ),
            tool_version="1.0.0",
            input_types={
                "prompt": "str - The natural language search query to guide the web agent.",
            },
            output_type="str - A reasoned and synthesized response based on web tool outputs.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(prompt="Compare the latest MacBook and Dell XPS specs and recommend one.")',
                    "description": "Demonstrates reasoning: searches the web, extracts specs, compares, and makes a recommendation."
                },
            ],
        )
        self.llm_engine = create_llm_engine(model_string="agent-Qwen/Qwen2.5-7B-Instruct", is_multimodal=False) if model_string else None
        self.llm_engine.stop = ["</answer>", "</search>", "</search>\n", "</search>\n\n"]
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  
        self.turn = 2 
        self.google_tool = Google_Search() 

    def parse_query(self, content):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(content)
        if matches:
            return matches[-1]
        else:
            return None

    def search(self, query):
        result = self.google_tool.execute(query=query, num_results=10) # TODO: update tool
        return result

    def parse_answer(self, response):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(response)
        if matches:
            return matches[-1]
        else:
            return None

    def execute(self, prompt):
        if not self.llm_engine:
            return "Error: LLM engine not initialized. Please provide a valid model_string."
        try:        
            user_prompt = REASONING_SUMMARY_PREFIX + f" {prompt}\n"
            messages = [
                {"role": "user", "content": user_prompt},
            ]
            for i in range(self.turn + 1):
                print(f"Turn {i}", messages, "\n\n")
                if i == self.turn:
                    self.llm_engine.stop = ["</answer>"]
                completion = self.llm_engine(messages)
                role, content = completion.choices[0].message.role, completion.choices[0].message.content
                
                finish_reason = completion.choices[0].finish_reason
                stop_reason = completion.choices[0].stop_reason
                if finish_reason == "stop" and stop_reason is None:
                    messages.append({"role": role, "content": content})
                    break
                elif finish_reason == "stop" and stop_reason == "</answer>":
                    content = content + "</answer>"
                    messages.append({"role": role, "content": content})
                    break
                elif finish_reason == "stop" and stop_reason == "</search>":
                    content = content + "</search>"
                    messages.append({"role": role, "content": content})
                    query = self.parse_query(content)
                    result = self.search(query) if query else None 
                    messages.append({"role": "tool", "content": f"<information>{result}</information>"})
            print("Whole messages \n", messages)
            response = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            # answer = response
            answer = self.parse_answer(response)
            print("[DEBUG] Return Response: ", answer)
            return answer if answer else "No results found."

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['require_llm_engine'] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata

if __name__ == "__main__":  
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/web_agent
    python tool.py
    """

    import json

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Web_Agent_Tool
    tool = Web_Agent_Tool("Qwen/Qwen2.5-7B-Instruct")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(json.dumps(metadata, indent=4))

    # Test queries
    test_queries = [
        # "Compare the latest MacBook and Dell XPS specs and recommend one.",
        # "What are the top 3 tourist attractions in Tokyo?",
        # "What rocket was used to launch the first spacecraft that approached Uranus?"
        # "Voyager 2 launch vehicle"
        # "In what year was Best Buy added to the S&P 500 index?"
        # "In what year was the company that was founded as Sound of Music added to the S&P 500?"
        "In what year was the tallest fixed steel structure completed?"
    ]

    # Execute the tool with test queries
    for query in test_queries:
        print(f"\nTesting query: {query}")
        try:
            execution = tool.execute(prompt=query)
            print("Generated Response:")
            print(json.dumps(execution, indent=4))
        except Exception as e:
            print(f"Execution failed: {e}")

    print("\nDone!")

""" 
python octotools/tools/web_agent/tool.py
"""