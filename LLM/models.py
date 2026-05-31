"""
Model configurations for code summarization
Supports: CodeBERT, GraphCodeBERT, CodeT5, PLBART, UniXcoder,
          StarCoder2, DeepSeek-Coder, CodeGemma, Qwen2.5-Coder, Phi-3.5, CodeLlama
"""

from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    RobertaConfig,
    EncoderDecoderModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PLBartForConditionalGeneration,
    PLBartTokenizer,
)
from peft import LoraConfig


class ModelConfig:
    """Base model configuration"""
    
    def __init__(self, model_name, use_lora=False):
        self.model_name = model_name
        self.use_lora = use_lora
        # Ablation - Fixed: Longer context for code summarization
        self.max_source_length = 4096 
        self.max_target_length = 128
        self.is_causal_lm = False  # True for decoder-only models
    
    def get_tokenizer(self):
        raise NotImplementedError
    
    def get_model(self, quantization_config=None):
        raise NotImplementedError
    
    def get_lora_config(self):
        """Default LoRA configuration"""
        return LoraConfig(
            r=16,
            lora_alpha=16, # FIXED: Set alpha equal to r for better numerical stability
            target_modules=self.get_lora_target_modules(),
            lora_dropout=0.05,
            bias="none",
            task_type=self.get_task_type()
        )
    
    def get_lora_target_modules(self):
        """Override in subclass"""
        return ["query", "value"]
    
    def get_task_type(self):
        """Override in subclass"""
        return "SEQ_2_SEQ_LM"
    
    # UPDATED SIGNATURE: Accept tokenizer
    def format_prompt(self, code, tokenizer=None):
        """Configure model for generation"""
        return code # Default pass-through
        
    def format_prompt_clone_detection(self, pair_data, tokenizer=None):
        """Configure model for clone detection generation"""
        return pair_data

    # --- NEW: Added for Code Classification Task ---
    def format_prompt_classification(self, method_data, tokenizer=None):
        """Configure model for multiclass project classification"""
        return method_data
    
    def configure_model_for_generation(self, model, tokenizer):
        """Configure model for generation"""
        pass


# ============================================================================
# Original Models (Encoder-Decoder)
# ============================================================================

class CodeBERTConfig(ModelConfig):
    """CodeBERT (microsoft/codebert-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("codebert", use_lora)
        self.base_model = "microsoft/codebert-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length
        model.config.min_length = 10
        model.config.no_repeat_ngram_size = 3


class GraphCodeBERTConfig(ModelConfig):
    """GraphCodeBERT (microsoft/graphcodebert-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("graphcodebert", use_lora)
        self.base_model = "microsoft/graphcodebert-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length


class CodeT5Config(ModelConfig):
    """CodeT5 (Salesforce/codet5-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("codet5", use_lora)
        self.base_model = "Salesforce/codet5-base"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = T5ForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q", "v", "k", "o", "wi", "wo"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.max_length = self.max_target_length
        model.config.min_length = 10
        model.config.no_repeat_ngram_size = 3


class PLBARTConfig(ModelConfig):
    """PLBART (uclanlp/plbart-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("plbart", use_lora)
        self.base_model = "uclanlp/plbart-base"
    
    def get_tokenizer(self):
        tokenizer = PLBartTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = PLBartForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = PLBartForConditionalGeneration.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.max_length = self.max_target_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]


class UniXcoderConfig(ModelConfig):
    """UniXcoder (microsoft/unixcoder-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("unixcoder", use_lora)
        self.base_model = "microsoft/unixcoder-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length


# ============================================================================
# Small Open-Source Code LLMs (Decoder-Only / Causal LM)
# ============================================================================
class CausalLMConfig(ModelConfig):
    """Base class for causal language models (decoder-only)"""
    
    def __init__(self, model_name, use_lora=False):
        super().__init__(model_name, use_lora)
        self.is_causal_lm = True
        self.max_source_length = 4096 # Alignment with base config
        self.max_target_length = 128
    
    def get_task_type(self):
        return "CAUSAL_LM"
    
    def format_prompt(self, code, tokenizer=None):
        """Default format prompt"""
        return f"Summarize the following code:\n\n{code}\n\nSummary:"
    
    def format_prompt_clone_detection(self, pair_data, tokenizer=None):
        """Generic fallback prompt for Code Clone Detection"""
        return f"Determine if the following two Java methods are code clones. Answer strictly with 'Yes' or 'No'.\n\n{pair_data}\n\nAnswer:"

    def format_prompt_classification(self, method_data, tokenizer=None):
        """Generic fallback prompt for Code Classification"""
        return (
            "Predict the open-source project for the following Java method.\n"
            "You must output exactly one of these 11 project names: [openjdk11, deeplearning4j, eclipse.jdt.core, "
            "guava, commons-math, freemind, commons-collections, caffeine, checkstyle, commons-lang, trove].\n\n"
            f"{method_data}\n\n### Project Name:"
        )
    
    # def format_prompt_vulnerability(self, method_data, tokenizer=None):
    #     """Generic fallback prompt for binary vulnerability detection"""
    #     return (
    #         "Analyze the following Java method and its context. Determine if the '### Source Code' is vulnerable.\n"
    #         "Output exactly 'Yes' if it is vulnerable, or 'No' if it is clean.\n\n"
    #         f"{method_data}\n\n"
    #         "### Vulnerable (Yes/No):"
    #     )

    # Inside class CausalLMConfig(ModelConfig):

    def format_prompt_vulnerability(self, method_data, tokenizer=None):
        """Generic adversarial fallback for binary vulnerability detection."""
        system_content = (
            "You are an aggressive Red Team Security Auditor. Your goal is to find "
            "even the most subtle vulnerabilities that could lead to a system compromise. "
            "Do not assume the code is safe; look for edge cases, resource leaks, and logic flaws.\n\n"
            "You will be provided with Source Code and additional context sections (Version History, Call Graph, Method Age)."
        )
        
        user_content = (
            f"Real Data to Audit:\n\n{method_data}\n\n"
            "Based on your audit, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
            "Constraints:\n"
            "1. Output exactly one word: 'Yes' or 'No'.\n"
            "2. Do NOT provide any explanation, preamble, or markdown formatting.\n\n"
            "### Is Vulnerable:"
        )
        return f"{system_content}\n\n{user_content}"

    def configure_model_for_generation(self, model, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

class StarCoder2Config(CausalLMConfig):
    """StarCoder2 (bigcode/starcoder2-7b / bigcode/starcoder2-3b)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"starcoder2-{size}", use_lora)
        self.size = size
        self.base_model = f"bigcode/starcoder2-{size}"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class DeepSeekCoderConfig(CausalLMConfig):
    """DeepSeek-Coder Instruct (deepseek-ai/deepseek-coder-6.7b-instruct)"""
    
    def __init__(self, use_lora=False, size="6.7b"):
        super().__init__(f"deepseek-coder-{size.lower()}", use_lora)
        self.size = size
        # CHANGED: Now explicitly pointing to the instruction-tuned models
        self.base_model = f"deepseek-ai/deepseek-coder-{size}-instruct"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # CHANGED: Required for batched zero-shot inference with causal models
        tokenizer.padding_side = "left" 
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def format_prompt(self, code, tokenizer=None):
        system_content = (
            "You are an expert technical documentation assistant. "
            "Your task is to analyze Java code and its surrounding context to understand its exact purpose.\n\n"
            "You may be provided with the following Context sections:\n"
            "- '### Source Code': The primary Java method to summarize.\n"
            "- '### Version History': Previous versions to understand logic evolution.\n"
            "- '### Caller Context': Methods that call this method.\n"
            "- '### Callee Context': Methods called by this method.\n"
            "- '### Method Age': The age of the method in days.\n\n"
        )
        
        user_content = (
            f"Here is the data:\n\n{code}\n\n"
            "Based on the information above, Write a SINGLE-sentence summary for the primary Java method.\n"
            "Constraints:\n"
            "1. Start directly with an action verb (e.g., 'Validates...', 'Parses...').\n"
            "2. Do NOT start with 'The method...' or 'This code...'.\n\n"
            "### Summary:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback specifically for DeepSeek's native prompt structure
            return (
                f"{system_content}\n"
                f"### Instruction:\n{user_content}\n"
                f"### Response:\n"
            ).strip()
    
    def format_prompt_clone_detection(self, pair_data, tokenizer=None):
        """
        Formats the prompt for a Code Clone Detection task.
        
        Args:
            pair_data (str): The pre-formatted string containing Method 1, Method 2, 
                             and their respective context sections.
            tokenizer: The tokenizer for the specific LLM (e.g., CodeLlama, DeepSeek, Qwen).
        """
        system_content = (
            "You are an expert software engineering assistant specializing in code analysis. "
            "Your task is to analyze two Java methods (Method 1 and Method 2) and their "
            "surrounding context to determine if they are code clones (functionally identical or highly similar).\n\n"
            "You may be provided with the following Context sections for either or both methods:\n"
            "- '### Method 1 Source Code' and '### Method 2 Source Code': Two primary Java methods.\n"
            "- '### Method 1 Version History' and '### Method 2 Version History': Previous versions to understand logic evolution.\n"
            "- '### Method 1 Caller Context' and '### Method 2 Caller Context': Methods that call this method.\n"
            "- '### Method 1 Callee Context' and '### Method 2 Callee Context': Methods called by this method.\n"
            "- '### Method 1 Age' and '### Method 2 Age': The age of the method in days.\n\n"
        )
        
        user_content = (
            f"Here is the data for Method 1 and Method 2:\n\n{pair_data}\n\n"
            "Based on the information above, determine if the two Java methods are code clones. "
            "Answer strictly with 'Yes' or 'No'.\n"
            "Constraints:\n"
            "1. Output exactly one word: 'Yes' or 'No'.\n"
            "2. Do NOT provide any explanation, preamble, or markdown formatting.\n\n"
            "### Is Clone:"
        )
        
        # Leverages the tokenizer's built-in template for cross-model compatibility
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback specifically for Llama's native [INST] <<SYS>> prompt structure
            return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"

    def format_prompt_classification(self, method_data, tokenizer=None):
        """Formats the prompt for the Code Classification task."""
        system_content = (
            "You are an expert software engineering assistant specializing in code classification.\n"
            "Your task is to analyze a single Java method and its surrounding context to predict which open-source project it belongs to.\n\n"
            "You must output exactly one of the following 11 project names: [openjdk11, deeplearning4j, eclipse.jdt.core, guava, commons-math, freemind, commons-collections, caffeine, checkstyle, commons-lang, trove].\n"
            "Do not output any other text, explanation, or markdown formatting."
        )
        
        user_content = (
            f"Here is the data for the method:\n\n{method_data}\n\n"
            "Based on the information above, predict the exact project name from the allowed list.\n"
            "### Project Name:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return f"{system_content}\n### Instruction:\n{user_content}\n### Response:\n".strip()
    
    def format_prompt_vulnerability(self, method_data, tokenizer=None):
        system_content = (
            "You are an expert software engineering assistant specializing in secure code analysis. "
            "Your task is to analyze a Java method and its surrounding context to determine if it "
            "contains a security vulnerability.\n\n"
            "Context sections provided may include:\n"
            "- '### Source Code': The primary Java method.\n"
            "- '### Version History': Historical iterations of the method code.\n"
            "- '### Caller Context': Method(s) that call this target method.\n"
            "- '### Callee Context': Method(s) that this target method invokes.\n"
            "- '### Method Age': Total lifespan of the method in days.\n\n"
        )
        
        user_content = (
            f"Analyze the following method data:\n\n{method_data}\n\n"
            "Based on the analysis, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
            "Constraints:\n"
            "1. Output strictly one word: 'Yes' or 'No'.\n"
            "2. No preamble or explanation.\n\n"
            "### Is Vulnerable:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"### Instruction:\n{system_content}\n{user_content}\n### Response:\n"

    def format_prompt_vulnerability(self, method_data, tokenizer=None):
        """
        Formats the prompt for DeepSeek-Coder with an Adversarial Persona 
        and full context explanations.
        """
        system_content = (
            "You are an aggressive Red Team Security Auditor. Your task is to perform "
            "a rigorous security analysis of Java methods. You must be highly critical "
            "and look for subtle logic flaws, resource leaks, and injection points. "
            "Do not assume the code is safe; assume it is suspicious until proven otherwise.\n\n"
            "You may be provided with the following Context sections:\n"
            "- '### Source Code': The primary Java method to analyze for vulnerabilities.\n"
            "- '### Version History': Previous versions of this method to understand logic evolution.\n"
            "- '### Caller Context' and '### Callee Context': Methods that call or are called by this method.\n"
            "- '### Method Age': The age of the method in days since its first commit.\n\n"
        )
        
        user_content = (
            f"Analyze the following method data for potential security flaws:\n\n{method_data}\n\n"
            "Based on your audit, is the '### Source Code' vulnerable? Answer strictly with 'Yes' or 'No'.\n"
            "Constraints:\n"
            "1. Output exactly one word: 'Yes' or 'No'.\n"
            "2. Do NOT provide any explanation, preamble, or markdown formatting.\n\n"
            "### Is Vulnerable:"
        )
        
        # Use the model's official template if available
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content}, 
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Specialized DeepSeek instruction fallback
            return f"### Instruction:\n{system_content}\n{user_content}\n### Response:\n"

    # def format_prompt_vulnerability(self, method_data, tokenizer=None):
    #     """DeepSeek-Coder: Adversarial Persona + 2-Shot (Vulnerable & Clean)"""
    #     system_content = (
    #         "You are an aggressive Red Team Security Auditor. Your task is to perform "
    #         "a rigorous security analysis of Java methods. You must be highly critical, "
    #         "assuming the code is suspicious until proven otherwise.\n\n"
    #         "Context Glossary:\n"
    #         "- '### Source Code': The primary Java method to analyze.\n"
    #         "- '### Version History': Previous versions to understand logic evolution.\n"
    #         "- '### Caller Context' and '### Callee Context': Structural dependencies.\n"
    #         "- '### Method Age': Lifespan of the method in days.\n\n"
    #     )
        
    #     # 2-Shot Anchor defining the decision boundary
    #     few_shot_examples = (
    #         "### Example 1 (Vulnerable)\n"
    #         "### Source Code: public void run(String p) { Runtime.getRuntime().exec(p); }\n"
    #         "### Is Vulnerable: Yes\n\n"
    #         "### Example 2 (Clean)\n"
    #         "### Source Code: public void run(String p) { if(isValid(p)) { Runtime.getRuntime().exec(p); } }\n"
    #         "### Is Vulnerable: No\n\n"
    #     )
        
    #     user_content = (
    #         f"{few_shot_examples}"
    #         f"### Real Data to Audit:\n{method_data}\n\n"
    #         "Based on your audit, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
    #         "### Is Vulnerable:"
    #     )
        
    #     if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
    #         messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    #         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     return f"### Instruction:\n{system_content}\n{user_content}\n### Response:\n"
        
class CodeGemmaConfig(CausalLMConfig):
    """CodeGemma (google/codegemma-7b / google/codegemma-2b)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"codegemma-{size}", use_lora)
        self.size = size
        self.base_model = f"google/codegemma-{size}"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]

class Qwen25CoderConfig(CausalLMConfig):
    """Qwen2.5-Coder (1.5B / 7B / 14B / 32B)"""
    
    def __init__(self, use_lora=False, size="7B"):
        # Normalizing size to handle both "14B" and "14b" inputs from Registry
        super().__init__(f"qwen25-coder-{size.lower()}", use_lora)
        self.size = size
        self.base_model = f"Qwen/Qwen2.5-Coder-{size}-Instruct"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.padding_side = "left" 
        return tokenizer
    
    def get_model(self, quantization_config=None):
        # Qwen2.5 models benefit from trust_remote_code for optimized attention kernels
        load_kwargs = {
            "pretrained_model_name_or_path": self.base_model,
            "device_map": "auto",
            "trust_remote_code": True
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)
    
    def get_lora_target_modules(self):
        # Standard Qwen-2.5 attention and MLP layers
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def format_prompt_clone_detection(self, pair_data, tokenizer=None):
        """
        Formats the prompt for a Code Clone Detection task.
        Optimized for Qwen's ChatML format.
        """
        system_content = (
            "You are an expert software engineering assistant. Your task is to analyze "
            "two Java methods and their surrounding context to determine if they are "
            "code clones (functionally identical or highly similar).\n\n"
            "Evaluate the Source Code, Version History, and Call Context provided."
        )
        
        user_content = (
            f"Here is the data for Method 1 and Method 2:\n\n{pair_data}\n\n"
            "Based on the information above, are these methods code clones? "
            "Answer strictly with 'Yes' or 'No'.\n"
            "### Is Clone:"
        )
        
        # 1. Primary path: Use the model's official ChatML template
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 2. Correct Fallback: Qwen uses ChatML (<|im_start|>) NOT Llama's [INST]
            return (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ).strip()

    def format_prompt_classification(self, method_data, tokenizer=None):
        """Formats the prompt for the Code Classification task using Qwen's optimal structure."""
        system_content = (
            "You are an expert software engineering assistant specializing in code classification.\n"
            "Your task is to analyze a single Java method and its surrounding context to predict which open-source project it belongs to.\n\n"
            "You must output exactly one of the following 11 project names: [openjdk11, deeplearning4j, eclipse.jdt.core, guava, commons-math, freemind, commons-collections, caffeine, checkstyle, commons-lang, trove].\n"
            "Do not output any other text, explanation, or markdown formatting."
        )
        
        user_content = (
            f"Here is the data for the method:\n\n{method_data}\n\n"
            "Based on the information above, predict the exact project name from the allowed list.\n"
            "### Project Name:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ).strip()
        
    # def format_prompt_vulnerability(self, method_data, tokenizer=None):
    #     system_content = (
    #         "You are an expert software engineering assistant specializing in vulnerability detection. "
    #         "Evaluate the Java method provided alongside its structural and temporal context.\n\n"
    #         "Context Glossary:\n"
    #         "- '### Source Code': The current code under evaluation.\n"
    #         "- '### Version History': Previous method states.\n"
    #         "- '### Caller/Callee Context': Graph-based structural context.\n"
    #         "- '### Method Age': Temporal signal in days.\n\n"
    #     )
        
    #     user_content = (
    #         f"Data to analyze:\n\n{method_data}\n\n"
    #         "Is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
    #         "### Is Vulnerable:"
    #     )
        
    #     if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
    #         messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    #         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     else:
    #         return (
    #             f"<|im_start|>system\n{system_content}<|im_end|>\n"
    #             f"<|im_start|>user\n{user_content}<|im_end|>\n"
    #             f"<|im_start|>assistant\n"
    #         ).strip()

    def format_prompt_vulnerability(self, method_data, tokenizer=None):
        """
        Formats the prompt for Qwen2.5-Coder with an Adversarial Persona 
        and the required Context Glossary.
        """
        system_content = (
            "You are an aggressive Red Team Security Auditor. Your mission is to perform "
            "a rigorous security analysis of Java methods. You must be highly critical, "
            "assuming the code is suspicious until proven otherwise. Look for subtle "
            "logic flaws, resource leaks, and edge-case vulnerabilities that standard "
            "auditors might miss.\n\n"
            "Context Glossary:\n"
            "- '### Source Code': The primary Java method currently under evaluation.\n"
            "- '### Version History': Previous method states, used to understand logic evolution and potential regressions.\n"
            "- '### Caller/Callee Context': Graph-based structural context showing interactions with other methods.\n"
            "- '### Method Age': Temporal signal indicating how many days the method has existed in the project.\n\n"
        )
        
        user_content = (
            f"Review the following data for potential security flaws:\n\n{method_data}\n\n"
            "Based on your audit, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
            "### Is Vulnerable:"
        )
        
        # 1. Primary path: Use Qwen's ChatML template for peak instruction-following
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content}, 
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 2. Manual Fallback: Explicit ChatML (<|im_start|>) structure
            return (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ).strip()

    # def format_prompt_vulnerability(self, method_data, tokenizer=None):
    #     """Qwen2.5-Coder: Adversarial Persona + 2-Shot + ChatML"""
    #     system_content = (
    #         "You are an aggressive Red Team Security Auditor. Your mission is to find "
    #         "subtle vulnerabilities in Java methods. Do not assume safety; look for flaws.\n\n"
    #         "Context Glossary:\n"
    #         "- '### Source Code': The current code under evaluation.\n"
    #         "- '### Version History': Historical iterations.\n"
    #         "- '### Caller/Callee Context': Graph-based context.\n"
    #         "- '### Method Age': Temporal signal in days.\n\n"
    #     )
        
    #     few_shot_examples = (
    #         "### Example 1 (Vulnerable)\n"
    #         "### Source Code: public int get(int i) { return array[i]; }\n"
    #         "### Is Vulnerable: Yes\n\n"
    #         "### Example 2 (Clean)\n"
    #         "### Source Code: public int get(int i) { if(i >= 0 && i < array.length) return array[i]; return -1; }\n"
    #         "### Is Vulnerable: No\n\n"
    #     )
        
    #     user_content = (
    #         f"{few_shot_examples}"
    #         f"### Real Data to Audit:\n{method_data}\n\n"
    #         "Is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
    #         "### Is Vulnerable:"
    #     )
        
    #     if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
    #         messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    #         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     return (
    #         f"<|im_start|>system\n{system_content}<|im_end|>\n"
    #         f"<|im_start|>user\n{user_content}<|im_end|>\n"
    #         f"<|im_start|>assistant\n"
    #     ).strip()

class Phi35MiniConfig(CausalLMConfig):
    """Phi-3.5-mini-instruct (microsoft/Phi-3.5-mini-instruct)"""
    
    def __init__(self, use_lora=False):
        super().__init__("phi-3.5-mini", use_lora)
        self.base_model = "microsoft/Phi-3.5-mini-instruct"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    
    def format_prompt(self, code):
        """Phi-3.5 uses specific chat format"""
        return f"<|user|>\nSummarize the following code:\n\n{code}<|end|>\n<|assistant|>\n"

class CodeLlamaConfig(CausalLMConfig):
    """CodeLlama Instruct (codellama/CodeLlama-7b-Instruct-hf / 13b / 34b)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"codellama-{size.lower()}", use_lora)
        self.size = size
        # CHANGED: Now explicitly pointing to the instruction-tuned models
        self.base_model = f"codellama/CodeLlama-{size}-Instruct-hf"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # CHANGED: Required for batched inference
        tokenizer.padding_side = "left" 
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def format_prompt(self, code, tokenizer=None):
        system_content = (
            "You are an expert technical documentation assistant. "
            "Your task is to analyze Java code and its surrounding context to understand its exact purpose.\n\n"
            "You may be provided with the following Context sections:\n"
            "- '### Source Code': The primary Java method to summarize.\n"
            "- '### Version History': Previous versions to understand logic evolution.\n"
            "- '### Caller Context': Methods that call this method.\n"
            "- '### Callee Context': Methods called by this method.\n"
            "- '### Method Age': The age of the method in days."
        )
        
        user_content = (
            f"Here is the data:\n\n{code}\n\n"
            "Based on the information above, Write a SINGLE-sentence summary for the primary Java method.\n"
            "Constraints:\n"
            "1. Start directly with an action verb (e.g., 'Validates...', 'Parses...').\n"
            "2. Do NOT start with 'The method...' or 'This code...'.\n\n"
            "### Summary:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback specifically for Llama's native [INST] <<SYS>> prompt structure
            return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"

    def format_prompt_clone_detection(self, pair_data, tokenizer=None):
        """
        Formats the prompt for a Code Clone Detection task.
        Specialized for CodeLlama-7B with 1-Shot anchoring and sandwiching.
        """
        # --------------------------------------------------------------------------
        # 1. 1-SHOT OPTIMIZED PROMPT FOR CODELLAMA-7B
        # --------------------------------------------------------------------------
        if "codellama-7b" in self.model_name.lower():
            # Minimalist 1-shot example of a clone
            example_clone = (
                "### Method 1 Source Code\npublic int add(int a, int b) { return a + b; }\n\n"
                "====================\n\n"
                "### Method 2 Source Code\npublic int sum(int x, int y) { return x + y; }"
            )
            
            system_7b = (
                "Task: Determine if two Java methods are code clones (functionally similar).\n"
                "Output exactly 'Yes' or 'No'."
            )
            
            user_7b = (
                f"Example 1 (Clone):\n{example_clone}\nIs Clone: Yes\n\n"
                f"Now analyze the following data:\n\n{pair_data}\n\n"
                "Are Method 1 and Method 2 code clones? Answer strictly with 'Yes' or 'No'.\n"
                "### Is Clone:"
            )
            
            return f"<s>[INST] <<SYS>>\n{system_7b}\n<</SYS>>\n\n{user_7b} [/INST]"

        # --------------------------------------------------------------------------
        # 2. ORIGINAL PROMPT FOR ALL OTHER MODELS (13B, DeepSeek, Qwen)
        # --------------------------------------------------------------------------
        system_content = (
            "You are an expert software engineering assistant specializing in code analysis. "
            "Your task is to analyze two Java methods (Method 1 and Method 2) and their "
            "surrounding context to determine if they are code clones (functionally identical or highly similar).\n\n"
            "You may be provided with the following Context sections for either or both methods:\n"
            "- '### Method 1 Source Code' and '### Method 2 Source Code': Two primary Java methods.\n"
            "- '### Method 1 Version History' and '### Method 2 Version History': Previous versions to understand logic evolution.\n"
            "- '### Method 1 Caller Context' and '### Method 2 Caller Context': Methods that call this method.\n"
            "- '### Method 1 Callee Context' and '### Method 2 Callee Context': Methods called by this method.\n"
            "- '### Method 1 Age' and '### Method 2 Age': The age of the method in days.\n\n"
        )
        
        user_content = (
            f"Here is the data for Method 1 and Method 2:\n\n{pair_data}\n\n"
            "Based on the information above, determine if the two Java methods are code clones. "
            "Answer strictly with 'Yes' or 'No'.\n"
            "Constraints:\n"
            "1. Output exactly one word: 'Yes' or 'No'.\n"
            "2. Do NOT provide any explanation, preamble, or markdown formatting.\n\n"
            "### Is Clone:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"

    def format_prompt_classification(self, method_data, tokenizer=None):
        """Formats the prompt for the Code Classification task natively for Llama with a 1-shot anchor."""
        system_content = (
            "You are an expert software engineering assistant specializing in code classification.\n"
            "Your task is to analyze a single Java method and its surrounding context to predict which open-source project it belongs to.\n\n"
            "You must output exactly one of the following 11 project names: [openjdk11, deeplearning4j, eclipse.jdt.core, guava, commons-math, freemind, commons-collections, caffeine, checkstyle, commons-lang, trove].\n"
            "Do not output any other text, explanation, or markdown formatting."
        )
        
        # Defining a clear 1-shot example
        example_shot = (
            "### Example 1\n"
            "### Source Code: public int add(int a, int b) { return a + b; }\n"
            "### Project Name: commons-math\n\n"
        )
        
        user_content = (
            f"{example_shot}"
            f"### Real Data to Classify:\n"
            f"{method_data}\n\n"
            "Based on the information above, predict the exact project name from the allowed list.\n"
            "### Project Name:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Standard Llama-2/3 Instruction Format
            return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"
    
    # def format_prompt_vulnerability(self, method_data, tokenizer=None):
    #     system_content = (
    #         "You are an expert software engineering assistant specializing in secure code analysis. "
    #         "Your task is to analyze a Java method and its surrounding context to determine if it is vulnerable.\n\n"
    #         "Context sections may include Source Code, Version History, Call Graph Context, and Method Age."
    #     )
        
    #     # 1-Shot Anchor to enforce strictly binary output
    #     example_shot = (
    #         "### Example 1\n### Source Code: public void test() { char[] c = new char[10]; c[11] = 'a'; }\n"
    #         "### Is Vulnerable: Yes\n\n"
    #     )
        
    #     user_content = (
    #         f"{example_shot}"
    #         f"### Real Data to Analyze:\n{method_data}\n\n"
    #         "Based on the provided information, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
    #         "### Is Vulnerable:"
    #     )
        
    #     if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
    #         messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    #         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     else:
    #         return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"       
    
    def format_prompt_vulnerability(self, method_data, tokenizer=None):
        """
        Formats the prompt for CodeLlama with an Adversarial Persona, 
        1-Shot Anchor, and full Context Glossary.
        """
        system_content = (
            "You are an aggressive Red Team Security Auditor. Your mission is to perform "
            "a rigorous security analysis of Java methods. You must be highly critical, "
            "assuming the code is suspicious until proven otherwise. Look for subtle "
            "logic flaws, resource leaks, and edge-case vulnerabilities that standard "
            "auditors might miss.\n\n"
            "Context Glossary:\n"
            "- '### Source Code': The primary Java method currently under evaluation.\n"
            "- '### Version History': Previous method states, used to understand logic evolution.\n"
            "- '### Caller Context' and '### Callee Context': Graph-based structural context showing method interactions.\n"
            "- '### Method Age': Temporal signal indicating how many days the method has existed.\n\n"
        )
        
        # 1-Shot Anchor to enforce strictly binary output and define the "Red Team" expectation
        example_shot = (
            "### Example 1\n"
            "### Source Code: public void test() { char[] c = new char[10]; c[11] = 'a'; }\n"
            "### Is Vulnerable: Yes\n\n"
        )
        
        user_content = (
            f"{example_shot}"
            f"### Real Data to Audit:\n{method_data}\n\n"
            "Based on the provided information, is the '### Source Code' vulnerable? Answer strictly 'Yes' or 'No'.\n"
            "### Is Vulnerable:"
        )
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Traditional Llama-2 [INST] format
            return f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"

class Qwen3CoderConfig(CausalLMConfig):
    """Qwen3 Coder (e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct)"""
    def __init__(self, use_lora=False, size="30B-A3B"):
        super().__init__(f"qwen3-coder-{size.lower()}-instruct", use_lora)
        self.size = size
        self.base_model = f"Qwen/Qwen3-Coder-{size}-Instruct"
    
    def get_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.padding_side = "left" 
        return tokenizer
    
    def get_model(self, quantization_config=None):
        from transformers import AutoModelForCausalLM
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        return model

    # --- ADD THIS NEW METHOD TO FIX THE LORA ERROR ---
    def get_lora_config(self):
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], # Correct Qwen3 architecture targets
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    # Original encoder-decoder models
    "codebert": CodeBERTConfig,
    "graphcodebert": GraphCodeBERTConfig,
    "codet5": CodeT5Config,
    "plbart": PLBARTConfig,
    "unixcoder": UniXcoderConfig,
    
    # Small open-source code LLMs (decoder-only)
    "starcoder2-3b": lambda use_lora=False: StarCoder2Config(use_lora, size="3b"),
    "starcoder2-7b": lambda use_lora=False: StarCoder2Config(use_lora, size="7b"),
    
    # Updated DeepSeek keys to explicitly show "instruct"
    "deepseek-coder-1.3b-instruct": lambda use_lora=False: DeepSeekCoderConfig(use_lora, size="1.3b"), 
    "deepseek-coder-6.7b-instruct": lambda use_lora=False: DeepSeekCoderConfig(use_lora, size="6.7b"),
    "deepseek-coder-33b-instruct": lambda use_lora=False: DeepSeekCoderConfig(use_lora, size="33b"),
    
    "codegemma-2b": lambda use_lora=False: CodeGemmaConfig(use_lora, size="2b"),
    "codegemma-7b": lambda use_lora=False: CodeGemmaConfig(use_lora, size="7b"),
    "qwen25-coder-1.5b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="1.5B"),
    "qwen25-coder-7b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="7B"),
    "qwen25-coder-14b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="14B"),
    "qwen25-coder-32b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="32B"), 
    "qwen3-coder-30b-a3b-instruct": lambda use_lora=False: Qwen3CoderConfig(use_lora, size="30B-A3B"),
    "phi-3.5-mini": Phi35MiniConfig,
    
    # Updated CodeLlama keys to explicitly show "instruct"
    "codellama-7b-instruct": lambda use_lora=False: CodeLlamaConfig(use_lora, size="7b"),
    "codellama-13b-instruct": lambda use_lora=False: CodeLlamaConfig(use_lora, size="13b"),
    "codellama-34b-instruct": lambda use_lora=False: CodeLlamaConfig(use_lora, size="34b"),
}

def get_model_config(model_name, use_lora=False):
    """Factory function to get model configuration"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_REGISTRY.keys())}")
    
    config_class = MODEL_REGISTRY[model_name]
    if callable(config_class):
        return config_class(use_lora=use_lora)
    else:
        return config_class(use_lora=use_lora)

def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())

def get_model_info():
    """Get information about all models (Corrected for 14B+)"""
    info = {
        "encoder_decoder": [
            "codebert", "graphcodebert", "codet5", "plbart", "unixcoder"
        ],
        "decoder_only_small": {
            "1-3B": ["qwen25-coder-1.5b", "codegemma-2b", "starcoder2-3b", "deepseek-coder-1.3b-instruct"],
            "3-7B": ["phi-3.5-mini", "deepseek-coder-6.7b-instruct", "codegemma-7b", 
                     "codellama-7b-instruct", "qwen25-coder-7b", "starcoder2-7b"],
            "10B+": [
                "codellama-13b-instruct", "codellama-34b-instruct", 
                "deepseek-coder-33b-instruct", 
                "qwen25-coder-14b", "qwen25-coder-32b"
            ] 
        }
    }
    return info