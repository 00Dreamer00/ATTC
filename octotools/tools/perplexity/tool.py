import os
from octotools.tools.base import BaseTool
from octotools.engine.factory import create_llm_engine

class Perplexity_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string): 
        super().__init__(
            tool_name="Perplexity_Tool", 
            tool_description=(
        "A web agent tool that integrates search tool with reasoning ability "
        "to interpret user tasks, plan appropriate actions, autonomously invoke tool, and generate intelligent responses "
        "based on online information."
    ),
            tool_version="1.0.0",
            input_types={
                "prompt": "str - The natural language search query to guide the web agent.",
            },
            output_type="str - A reasoned and synthesized response based on web search outputs.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(prompt="Compare the latest MacBook and Dell XPS specs and recommend one.")',
                    "description": "Demonstrates reasoning: searches the web, extracts specs, compares, and makes a recommendation."
                },
            ],
        )
        self.llm_engine = create_llm_engine(model_string=model_string, is_multimodal=False) if model_string else None 

    def execute(self, prompt="Describe this image in detail."):
        try:
            if not self.llm_engine:
                return "Error: LLM engine not initialized. Please provide a valid model_string."
                
            caption = self.llm_engine(prompt)
            return caption
        except Exception as e:
            return f"Error generating search results: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['require_llm_engine'] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/perplexity
    python tool.py
    """

    import json

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Web_Agent_Tool
    tool = Perplexity_Tool(model_string="sonar-pro")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(json.dumps(metadata, indent=4))

    # Test queries
    test_queries = [
        "Who is the father of the father of George Washington?"
        # "Compare the latest MacBook and Dell XPS specs and recommend one.",
        # "What are the top 3 tourist attractions in Tokyo?",
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
