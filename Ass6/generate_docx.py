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
    p_ps.add_run("Design and implement a Convolutional Neural Network (CNN) for image classification using the Mango Leaf & Fruit Pathology dataset (6 classes: Anthracnose, Bacterial Canker, Die Back, Powdery Mildew, Sooty Mould, Healthy). Perform spatial data augmentation, evaluate model performance using accuracy/loss curves, confusion matrix, and classification report.")

    # Course Objectives & Relevance
    doc.add_heading("1. Course Objectives & Relevance", level=2)
    doc.add_paragraph(
        "1. Understand the core principles of Convolutional Neural Networks (CNNs) for visual pattern recognition.\n"
        "2. Implement 2D Convolution, Max Pooling, Batch Normalization, and Dropout layers in Keras.\n"
        "3. Design a data augmentation pipeline (rotation, flipping, zooming) to prevent overfitting on leaf disease textures.\n"
        "4. Evaluate multi-class classification accuracy, loss convergence, and class-wise precision/recall on a unique agricultural pathology dataset."
    )

    # Theoretical Foundations
    doc.add_heading("2. Theoretical Foundations & Mathematical Formulations", level=2)
    doc.add_paragraph(
        "A. 2D Convolution Operation:\n"
        "   Z(i, j) = sum_m sum_n (X(i*s + m, j*s + n) * K(m, n)) + b\n"
        "   Output Dimensions: H_out = floor((H_in - k + 2p)/s) + 1, W_out = floor((W_in - k + 2p)/s) + 1\n\n"
        "B. Max Pooling Operation:\n"
        "   P(i, j) = max_{0<=m<2, 0<=n<2} X(2i+m, 2j+n)\n\n"
        "C. Batch Normalization:\n"
        "   z_hat = (z - mu_B) / sqrt(sigma_B^2 + epsilon), y = gamma * z_hat + beta\n\n"
        "D. Categorical Cross-Entropy Loss:\n"
        "   L(y, y_hat) = - sum_{i=1}^{6} y_i * log(y_hat_i)"
    )

    # Step-by-Step Algorithm
    doc.add_heading("3. Algorithm / Step-by-Step Execution Plan", level=2)
    doc.add_paragraph(
        "1. Import TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Sklearn.\n"
        "2. Load and preprocess the Mango Leaf & Fruit Pathology dataset (1,200 images, 6 classes).\n"
        "3. Normalize image pixel intensity values from [0, 255] to [0.0, 1.0].\n"
        "4. One-hot encode targets into 6-dimensional class vectors.\n"
        "5. Define a Sequential CNN Architecture: Input(128, 128, 3) -> Data Augmentation -> Conv2D(32) + BatchNorm -> Conv2D(64) + BatchNorm -> Conv2D(128) + BatchNorm -> Conv2D(256) + Dropout(0.25) -> Flatten -> Dense(256, ReLU) -> Dense(6, Softmax).\n"
        "6. Calculate trainable parameters for each layer.\n"
        "7. Compile model using Adam optimizer (learning_rate = 0.001) and categorical_crossentropy loss.\n"
        "8. Train CNN for 25 epochs with batch size 32.\n"
        "9. Plot training vs validation loss and accuracy curves.\n"
        "10. Generate Confusion Matrix heatmap and Classification Report on test set.\n"
        "11. Record analytical conclusions."
    )

    # Summary Table: CNN Architecture
    doc.add_heading("4. CNN Model Architecture & Parameter Breakdown", level=2)
    table = doc.add_table(rows=8, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Layer Name", "Layer Type", "Output Shape", "Trainable Parameters"]
    for j, text in enumerate(headers):
        cell = table.cell(0, j)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    row_data = [
        ("Input Augmentation", "Flip / Rotation / Zoom", "(128, 128, 3)", "0"),
        ("Conv Block 1", "Conv2D(32) + BatchNorm + MaxPool", "(64, 64, 32)", "992 + 128 = 1,120"),
        ("Conv Block 2", "Conv2D(64) + BatchNorm + MaxPool", "(32, 32, 64)", "18,496 + 256 = 18,752"),
        ("Conv Block 3", "Conv2D(128) + BatchNorm + MaxPool", "(16, 16, 128)", "73,856 + 512 = 74,368"),
        ("Conv Block 4", "Conv2D(256) + Dropout + MaxPool", "(8, 8, 256)", "295,168"),
        ("Dense Layer", "Dense(256) + Dropout", "(256)", "4,194,560"),
        ("Output Layer", "Dense(6, Softmax)", "(6)", "1,542")
    ]

    for i, data in enumerate(row_data, start=1):
        for j, text in enumerate(data):
            cell = table.cell(i, j)
            cell.paragraphs[0].text = text
            if i % 2 == 1:
                set_cell_background(cell, "F2F2F2")

    doc.add_paragraph("\n")

    # Conclusion
    doc.add_heading("5. Conclusions & Summary", level=2)
    doc.add_paragraph(
        "1. Designed and trained a custom 4-Block CNN for multi-class Mango leaf disease classification.\n"
        "2. Applied spatial data augmentation to enhance feature invariance across varying lesion angles and scale.\n"
        "3. Batch normalization and dropout layers prevented overfitting, resulting in smooth convergence across 25 epochs.\n"
        "4. Generated complete evaluation metrics including Confusion Matrix, Loss/Accuracy curves, and Classification Report."
    )

    doc.save(output_path)
    print(f"Successfully generated Word document: {output_path}")

if __name__ == '__main__':
    create_submission_docx()
