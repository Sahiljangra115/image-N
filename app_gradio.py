"""
Gradio UI for the multimodal RAG: search images by text, get a grounded answer,
and generate an edited image from the retrieved one.

Run locally:   python app_gradio.py   (opens http://127.0.0.1:7860)
On HF Spaces:  this file is the entrypoint (Dockerfile CMD).

Heavy models (CLIP, SmolVLM, SD Turbo) load lazily on first use, so the page
appears instantly and the first query is the slow one.
"""

import gradio as gr

import image_rag
import multimodal

IMAGE_FOLDER = "data/images"


# ---------- handlers ----------

def search(query, mode):
    """Route by mode. Returns (answer, gallery_images, text_sources_md)."""
    if not query or not query.strip():
        return "Type a question first.", [], ""
    if mode == "Images only":
        answer, img = image_rag.answer_with_image(IMAGE_FOLDER, query)
        return answer, [img], ""
    # Both: text chunks + image, answered by SmolVLM (no Ollama dependency)
    answer, chunks, imgs = multimodal.answer_multimodal(query)
    return answer, imgs, "\n\n---\n\n".join(chunks)


def generate(query, edit, strength, steps):
    """Find the image for query, edit it per the prompt. Returns (src, out)."""
    if not query or not query.strip():
        raise gr.Error("Enter what image to find.")
    if not edit or not edit.strip():
        raise gr.Error("Enter how to edit it.")
    paths, index = image_rag.build_image_index(IMAGE_FOLDER)
    src = image_rag.retrieve_images(query, paths, index, k=1)[0]
    out = image_rag.modify_image(src, edit, strength=strength, steps=int(steps))
    return src, out


# ---------- UI ----------

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Multimodal RAG") as demo:
    gr.Markdown(
        "# Multimodal RAG\n"
        "Find an image by describing it, get an answer grounded in that image, "
        "or generate a new image edited from the one we found."
    )

    with gr.Tab("Search & Answer"):
        with gr.Row():
            with gr.Column(scale=2):
                q = gr.Textbox(label="Your question", placeholder="e.g. two children on a skateboard")
                mode = gr.Radio(
                    ["Both", "Images only"],
                    value="Both", label="Source",
                    info="Both grounds the answer in the text docs AND the retrieved image.",
                )
                go = gr.Button("Search", variant="primary")
                gr.Examples(
                    ["two dogs running on the beach", "a busy street market",
                     "which cake is nut free"],
                    inputs=q,
                )
            with gr.Column(scale=3):
                answer = gr.Textbox(label="Answer", lines=4, show_copy_button=True)
                gallery = gr.Gallery(label="Retrieved image(s)", columns=2, height=320)
                with gr.Accordion("Text sources", open=False):
                    sources = gr.Markdown()
        go.click(search, [q, mode], [answer, gallery, sources])

    with gr.Tab("Generate"):
        gr.Markdown(
            "Retrieval conditions generation: we find the image for your query, "
            "then SD Turbo edits **that** image. (Slow on CPU.)"
        )
        with gr.Row():
            with gr.Column():
                gq = gr.Textbox(label="Find image", placeholder="e.g. a street market")
                edit = gr.Textbox(label="Edit prompt", placeholder="e.g. snowy night, cinematic, detailed")
                strength = gr.Slider(0.1, 0.9, value=0.6, step=0.05, label="Edit strength",
                                     info="Low keeps the source; high changes more.")
                steps = gr.Slider(1, 4, value=4, step=1, label="Steps")
                gen = gr.Button("Generate", variant="primary")
            with gr.Column():
                src_img = gr.Image(label="Source (retrieved)", height=300)
                out_img = gr.Image(label="Generated", height=300)
        gen.click(generate, [gq, edit, strength, steps], [src_img, out_img])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
