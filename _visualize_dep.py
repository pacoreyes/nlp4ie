import spacy
from lib.visualizations import plot_dependency_visualization

nlp = spacy.load("en_core_web_trf")

sentence = "Employers are really, really happy."
output_path = "images/dependency_visualization.svg"
print("Plotting dependency visualization...")
plot_dependency_visualization(sentence, output_path, nlp)
print("Done!")