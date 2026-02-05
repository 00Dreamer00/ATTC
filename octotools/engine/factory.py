from typing import Any

def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = True, **kwargs) -> Any:
    """
    Factory function to create appropriate LLM engine instance.
    """
    if any(x in model_string for x in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "claude" in model_string:
        from .anthropic import ChatAnthropic
        return ChatAnthropic(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "sonar" in model_string:
        from .perplexity import ChatPerplexity
        return ChatPerplexity(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "agent" in model_string:
        from .vllm_tool import ChatToolVLLM
        model_string = model_string.replace("agent-", "")
        return ChatToolVLLM(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "action" in model_string:
        from .vllm_action import ChatActionVLLM
        model_string = model_string.replace("action-", "")
        return ChatActionVLLM(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "dashscope" in model_string:
        from .qwen import ChatQwen
        model_string = model_string.replace("dashscope-", "")
        return ChatQwen(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif any(x in model_string for x in ["deepseek-chat", "deepseek-reasoner"]):
        from .deepseek import ChatDeepseek
        return ChatDeepseek(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "gemini" in model_string:
        from .gemini import ChatGemini
        return ChatGemini(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "grok" in model_string:
        from .xai import ChatGrok
        return ChatGrok(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "vllm" in model_string:
        from .vllm import ChatVLLM
        model_string = model_string.replace("vllm-", "")
        return ChatVLLM(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "together" in model_string:
        from .together import ChatTogether
        model_string = model_string.replace("together-", "")
        return ChatTogether(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    else:
        raise ValueError(f"Engine {model_string} not supported. If you are using Together models, please ensure have the prefix 'together-' in the model string. If you are using VLLM models, please ensure have the prefix 'vllm-' in the model string. For other custom engines, you can edit the factory.py file and add its interface file to add support for your engine. Your pull request will be warmly welcomed!")
    