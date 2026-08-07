import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_submission_docx():
    output_path = r'd:\Vit\Academics Sem-5\Deep Learning (DL)\Lab\Ass6\60_Assignment-6_DL.docx'
    doc = docx.Document()

    # Document Header
    p_head = doc.add_paragraph()
    p_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_head.add_run("Vishwakarma Institute of Technology, Pune\n")
    run_title.bold = True
    run_title.font.size = Pt(16)
    run_sub = p_head.add_run("Department of Computer Engineering (Artificial Intelligence)\nCI3201: Deep Learning - Lab Assignment 6\n")
    run_sub.font.size = Pt(13)
    run_sub.bold = True
    
    doc.add_paragraph("="*80)

    # Problem Statement
    p_ps = doc.add_paragraph()
    r_ps_lbl = p_ps.add_run("Problem Statement: ")
    r_ps_lbl.bold = True
    r_ps_lbl.font.size = Pt(11)
    p_ps.add_run("Design and implement a Convolutional Neural Network (CNN) for multi-class image classification using the Soybean Leaf Disease dataset across 7 distinct pathology categories. Perform data normalization, data augmentation, parameter calculation, model training, and performance evaluation using a confusion matrix and classification report.")

    # Objectives
    doc.add_heading("1. Course Objectives & Relevance", level=2)
    doc.add_paragraph(
        "1. Understand the architecture of Convolutional Neural Networks (CNNs) for image processing.\n"
        "2. Implement 2D Convolution operations, Pooling layers, and Dropout regularization using Keras.\n"
        "3. Apply data augmentation (flipping, rotation, zoom) to handle dataset variance and prevent overfitting.\n"
        "4. Calculate output feature map dimensions and trainable parameters per layer.\n"
        "5. Evaluate classification accuracy across multi-class plant disease categories using confusion matrices."
    )

    # Theoretical Foundations
    doc.add_heading("2. Theoretical Foundations & Mathematical Equations", level=2)
    doc.add_paragraph(
        "A. 2D Convolution Operation:\n"
        "   S(i,j) = (I * K)(i,j) = sum_m sum_n I(i-m, j-n) K(m,n)\n\n"
        "B. Feature Map Dimension Calculation:\n"
        "   W_out = floor((W_in - F + 2P) / S) + 1\n"
        "   H_out = floor((H_in - F + 2P) / S) + 1\n\n"
        "C. Max Pooling Operation:\n"
        "   P(i,j) = max_{m,n} A(i*S + m, j*S + n)\n\n"
        "D. Softmax Activation & Categorical Cross-Entropy Loss:\n"
        "   y_hat_c = exp(z_c) / sum_k(exp(z_k))\n"
        "   L(y, y_hat) = - sum_c(y_c * log(y_hat_c))"
    )

    # Algorithm
    doc.add_heading("3. Algorithm / Step-by-Step Execution Plan", level=2)
    doc.add_paragraph(
        "1. Import TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Sklearn.\n"
        "2. Define the 7 Soybean Disease categories (Bacterial Blight, Brown Spot, Frogeye Spot, Downy Mildew, Target Spot, Sudden Death, Healthy).\n"
        "3. Preprocess images: Resize to (128, 128, 3) and scale pixel intensities to [0.0, 1.0].\n"
        "4. Define a Data Augmentation pipeline using RandomFlip, RandomRotation, and RandomZoom.\n"
        "5. Construct a Sequential CNN Architecture:\n"
        "   - Conv Block 1: Conv2D(32, 3x3) + BatchNorm + MaxPool2D(2x2)\n"
        "   - Conv Block 2: Conv2D(64, 3x3) + BatchNorm + MaxPool2D(2x2)\n"
        "   - Conv Block 3: Conv2D(128, 3x3) + BatchNorm + MaxPool2D(2x2)\n"
        "   - Conv Block 4: Conv2D(256, 3x3) + Dropout(0.25) + MaxPool2D(2x2)\n"
        "   - Dense Head: Flatten() -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(7, Softmax)\n"
        "6. Calculate trainable parameters for all layers.\n"
        "7. Compile model using Adam optimizer (eta = 0.001) and Categorical Crossentropy loss.\n"
        "8. Train for 20 epochs with a batch size of 32.\n"
        "9. Plot Training vs Validation Loss and Accuracy curves.\n"
        "10. Generate Confusion Matrix heatmap and Classification Report on unseen test samples.\n"
        "11. Record analytical observations and conclusions."
    )

    # Summary Table
    doc.add_heading("4. Architecture & Performance Summary", level=2)
    
    p_t1 = doc.add_paragraph()
    p_t1.add_run("Table 1: Layer-wise CNN Specification & Parameter Calculation").bold = True
    table1 = doc.add_table(rows=6, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["Layer Type", "Output Shape", "Filter / Node Config", "Trainable Parameters"]
    for j, text in enumerate(headers1):
        cell = table1.cell(0, j)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    row_data1 = [
        ("Conv Block 1", "(128, 128, 32)", "32 filters (3x3), MaxPool(2x2)", "896 params"),
        ("Conv Block 2", "(64, 64, 64)", "64 filters (3x3), MaxPool(2x2)", "18,496 params"),
        ("Conv Block 3", "(32, 32, 128)", "128 filters (3x3), MaxPool(2x2)", "73,856 params"),
        ("Conv Block 4", "(16, 16, 256)", "256 filters (3x3), MaxPool(2x2)", "295,168 params"),
        ("Dense Classifier", "(7)", "Flatten(16384) -> Dense(256) -> Dense(7)", "4,196,615 params")
    ]

    for i, data in enumerate(row_data1, start=1):
        for j, text in enumerate(data):
            cell = table1.cell(i, j)
            cell.paragraphs[0].text = text
            if i % 2 == 1:
                set_cell_background(cell, "F2F2F2")

    doc.add_paragraph("\n")

    # Conclusion
    doc.add_heading("5. Conclusions", level=2)
    doc.add_paragraph(
        "1. The custom 4-block Convolutional Neural Network effectively classified 7 distinct Soybean Leaf Disease categories.\n"
        "2. Hierarchical feature learning enabled early layers to capture fine leaf edges and veins, while deeper layers recognized specific disease lesion shapes and discolorations.\n"
        "3. Data Augmentation (flipping, rotation, zoom) effectively mitigated overfitting on leaf background patterns.\n"
        "4. Batch Normalization stabilized gradient propagation across conv layers, accelerating convergence during training."
    )

    doc.save(output_path)
    print(f"Successfully generated Word document: {output_path}")

if __name__ == '__main__':
    create_submission_docx()
