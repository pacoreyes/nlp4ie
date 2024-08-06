import spacy
from lib.visualizations import plot_dependency_visualization

nlp = spacy.load("en_core_web_trf")

sentence = "It's not conducive to capital investment."
output_path = "images/dependency_visualization.svg"
print("Plotting dependency visualization...")
plot_dependency_visualization(sentence, output_path, nlp)
print("Done!")
print(f"Find the SVG file in {output_path}")
