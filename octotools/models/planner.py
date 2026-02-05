import os
import re
from PIL import Image
from typing import Dict, Any, List, Tuple
import json
import ast

from octotools.engine.factory import create_llm_engine
from octotools.models.memory import Memory
from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification
from octotools.default_prompts import SYSTEM_PROMPT, USER_PROMPT, SYSTEM_PROMPT_TOOL_N1, USER_PROMPT_TOOL_N1

class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, verbose: bool = False, **kwargs):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=True)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False)
        self.action_llm_engine = create_llm_engine(model_string=kwargs.get("action_llm_engine_name", llm_engine_name), is_multimodal=False)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: str = 4000) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""

        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        self.query_analysis = ""
        print(f"[DEBUG] query_analysis: {self.query_analysis}")
        return str(self.query_analysis).strip()

    def extract_context_tool_and_command(self, response: Any) -> Tuple[str, str]:
        # NOTE: currently, only one function call is supported

        def extract_tool_call(tool_call_str: str) -> str:
            pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(pattern, tool_call_str, flags=re.DOTALL)
            if not match:
                return None
            last_match = match.group(match.lastindex).strip()
            return last_match

        def parse_tool_call(tool_call_str: str) -> dict:
            """
            parse the tool call string into a dictionary. We support the following three tool call formats:
            1. '[Perplexity_Tool(prompt="Who is the father of Galileo Galilei?")]'
            2. "Perplexity_Tool(prompt="Who is the father of Galileo Galilei?")"
            3. "{'name': 'Perplexity_Tool', 'arguments': {'prompt': 'Who is the father of Galileo Galilei?'}}"
            """
            s = tool_call_str.strip()
            # remove "[ ]" from the beginning and end of the string
            s = re.sub(r'^\[|\]$', '', s)
            try:
                # 解析为 AST
                tree = ast.parse(s, mode='eval')
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Input is not a function call.")

                func_name = tree.body.func.id
                args_dict = {}

                for kw in tree.body.keywords:
                    # 把每个参数的值转为真实 Python 值
                    args_dict[kw.arg] = ast.literal_eval(kw.value)

                return {
                    "name": func_name,
                    "arguments": args_dict
                }

            except Exception as e:
                raise ValueError(f"Failed to parse function call: {e}")
        
        print(f"[DEBUG] Response: {response}") # TODO: remove this
        tool_call_str = extract_tool_call(response)
        if tool_call_str is None:
            return None, None
        try:
            function_calls = [json.loads(tool_call_str)]
        except json.JSONDecodeError:
            # parsing such function call: ```Perplexity_Tool(prompt="abc")```
            function_calls = [parse_tool_call(tool_call_str)] # TODO: update catch exception
        except Exception as e: 
            print(f"Response: {response}")
            print(f"Error extracting tool and command: {str(e)}")
            raise e # TODO: update here
            # return None, None

        print(f"[DEBUG] Function calls: {function_calls}") # TODO: remove this
        if len(function_calls[0]) == 0 or function_calls[0].get("name") == "":
            return None, None

        tool_list = []
        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            tool_list.append(name)
            arguments = func_call["arguments"]
            execution_list.append(
                f"execution = tool.execute({','.join([f'{k}={repr(v)}' for k,v in arguments.items()])})"
            )
            
        return tool_list[0], execution_list[0]

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:

        def normalize_tool_name(tool_name: str) -> str:
            # Normalize the tool name to match the available tools
            for tool in self.available_tools:
                if tool.lower() in tool_name.lower():
                    return tool
            return "No matched tool given: " + tool_name
        
        try:
            if isinstance(response, NextStep):
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)(?=\n\n|\Z)"
                
                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()
            tool_name = normalize_tool_name(tool_name)
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

        return context, sub_goal, tool_name
        
    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int) -> Any:
        # TODO: add more model prompts
        if "tool-n1-reason" in self.action_llm_engine.model_string:
            system_prompt = SYSTEM_PROMPT_TOOL_N1.format(
                question=question,
                image=image,
                query_analysis=query_analysis,
                available_tools=self.available_tools,
                toolbox_metadata=self.toolbox_metadata,
                memory=memory.get_actions(),
                step_count=step_count,
                max_step_count=max_step_count,
                remaining_steps=max_step_count - step_count
            )
            prompt_generate_next_step = USER_PROMPT_TOOL_N1.format(question=question)
        else:
            system_prompt = SYSTEM_PROMPT
            prompt_generate_next_step = USER_PROMPT.format(
                question=question,
                image=image,
                query_analysis=query_analysis,
                available_tools=self.available_tools,
                toolbox_metadata=self.toolbox_metadata,
                memory=memory.get_actions(),
                step_count=step_count,
                max_step_count=max_step_count,
                remaining_steps=max_step_count - step_count
            )
        next_step = self.action_llm_engine(prompt_generate_next_step, system_prompt=system_prompt)
        return next_step

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory) -> Any:
        image_info = self.get_image_info(image)

        prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation: 
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation: 
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""

        input_data = [prompt_memory_verification]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        stop_verification = self.llm_engine_mm(input_data, response_format=MemoryVerification)

        return stop_verification

    def extract_conclusion(self, response: Any) -> str:
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            # if match:
            #     conclusion = match.group(1).upper()
            #     if conclusion in ['STOP', 'CONTINUE']:
            #         return conclusion
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion
            
            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'

    def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        final_output = self.llm_engine_mm(input_data)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        final_output = self.llm_engine_mm(input_data)

        return final_output
    