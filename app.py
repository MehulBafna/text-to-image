import torch 
from PIL import Image 
from diffusers import StableDiffusionPipeline 
import streamlit as st 
# Function to generate image from text 
def generate_image_from_text(prompt, output_image_path="output_image.png"): 
    try: # Load the pre-trained Stable Diffusion model from Hugging Face 
        model_id = "CompVis/stable-diffusion-v1-4" 
        model_dtype = torch.float32 # Use float32 for CPU performance 
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=model_dtype) # Since Render doesn't support GPUs, force CPU 
        device = "cpu" 
        pipe = pipe.to(device) 
        # Optionally: disable safety checker and enable attention slicing to optimize CPU performance 
        pipe.safety_checker = lambda images, clip_input: (images, False) 
        pipe.enable_attention_slicing() # Generate the image from text prompt 
        image = pipe(prompt).images[0] # Save the generated image 
        image.save(output_image_path) 
        return image 
    except Exception as e: 
        print(f"Error generating image: {e}") 
        return None # Streamlit application 

def main(): 
    st.title("Text to Image Generator") 
    prompt = st.text_input("Enter your text prompt:") 
    if st.button("Generate Image"): 
        if prompt: 
            with st.spinner("Generating image... (It may take a while on CPU)"): 
                generated_image = generate_image_from_text(prompt) 
                if generated_image: 
                    st.image(generated_image, caption="Generated Image", use_column_width=True) 
                else: st.error("Image generation failed.") 
        else: st.error("Please enter a prompt.") 
if __name__ == "__main__": 
    main()
