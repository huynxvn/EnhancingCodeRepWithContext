from transformers import AutoModel, AutoTokenizer
import torch

# Replace 'model_name' with the model identifier from Hugging Face (e.g., 'bert-base-uncased')
model_name = 'uclanlp/plbart-base'

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)





# Save the model
model_save_path = 'checkpoint_11_100000.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Optionally, save the tokenizer as well
tokenizer_save_path = 'tokenizer'
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")


# Create a new model instance
model_loaded = AutoModel.from_pretrained(model_name)

# Load the state dict
model_loaded.load_state_dict(torch.load(model_save_path))

# Verify the model
print("Model loaded and verified successfully.")


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text2text-generation", model="uclanlp/plbart-base")

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text2text-generation", model="uclanlp/plbart-base")

print("OKAY")