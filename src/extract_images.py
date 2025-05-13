# Extract images from PDF using PyMuPDF
import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # GRAY or RGB
                pix.save(os.path.join(output_folder, f"page{page_index+1}_img{img_index+1}.png"))
            else:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(os.path.join(output_folder, f"page{page_index+1}_img{img_index+1}.png"))
