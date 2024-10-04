from PIL import Image
from diffusers import StableDiffusionPipeline
import streamlit as st
import torch
#import gunicorn

# Function to generate image from text
def generate_image_from_text(prompt, output_image_path="output_image.png"):
    # Load the pre-trained Stable Diffusion model from Hugging Face
    model_id = "CompVis/stable-diffusion-v1-4"  # You can also use other models
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Move the model to GPU (if available) for faster generation
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image from text prompt
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(prompt).images[0]

    # Save the generated image
    image.save(output_image_path)
    return image

# Streamlit application
def main():
    st.title("Text to Image Generator")

    prompt = st.text_input("Enter your text prompt:")
    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                generated_image = generate_image_from_text(prompt)
                st.image(generated_image, caption="Generated Image", use_column_width=True)
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    main()
