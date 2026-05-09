"""
Generate an HTML report from folder structure containing images.
Place images in subfolders under CandidateReport/ and run this script
to automatically generate Report.html with embedded images organized by folder.
"""

import os
import pandas as pd
from pathlib import Path

def generate_html_report(report_path='CandidateReport/', output_file='Report.html'):
    """
    Scan folder structure for images and generate an HTML report.
    
    Parameters:
    -----------
    report_path : str
        Path to the CandidateReport directory
    output_file : str
        Output HTML filename
    """
    
    base_path = Path(report_path)
    
    # Scan for subdirectories (these will be your categories)
    categories = {}
    
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir() and folder.name != '__pycache__':
            images = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if images:
                categories[folder.name] = images
    
    # Start building HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Detection Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #007bff;
            margin-top: 30px;
            padding: 10px;
            background-color: #e7f3ff;
            border-radius: 5px;
        }
        .image-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .image-item {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-item:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        .image-item img {
            width: 100%;
            height: auto;
            border-radius: 3px;
            margin-bottom: 10px;
        }
        .image-name {
            font-weight: bold;
            color: #333;
            word-break: break-word;
        }
        .count {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>Detection Report</h1>
"""
    
    # Add each category section
    for category, images in categories.items():
        html += f'    <h2>{category} <span class="count">{len(images)}</span></h2>\n'
        html += '    <div class="image-container">\n'
        
        for img_path in images:
            # Create relative path from report file to image
            rel_path = os.path.relpath(img_path, base_path)
            html += f'''        <div class="image-item">
            <img src="{rel_path}" alt="{img_path.stem}">
            <div class="image-name">{img_path.stem}</div>
        </div>
'''
        
        html += '    </div>\n'
    
    html += """</body>
</html>
"""
    
    # Write the HTML file
    output_path = base_path / output_file
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Report generated: {output_path}")
    print(f"✓ Total categories: {len(categories)}")
    for cat, imgs in categories.items():
        print(f"  - {cat}: {len(imgs)} images")


if __name__ == '__main__':
    generate_html_report()
