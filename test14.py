from collections import Counter

# Data as provided
data = Counter({'support_adj_emphatic': 484, 'oppose_verb_con': 472, 'support_verb_pro': 436, 'support_adj_positive': 402, 'oppose_adj_emphatic': 387, 'oppose_adv_emphatic': 365, 'support_adv_emphatic': 342, 'support_verb_positive': 225, 'support_verb_emphatic': 223, 'oppose_verb_con_pattern': 219, 'oppose_verb_emphatic': 213, 'oppose_adv_certainty': 187, 'support_verb_certainty': 179, 'oppose_adj_negative': 164, 'support_adv_certainty': 157, 'oppose_modality_predictive': 140, 'oppose_verb_pro': 139, 'oppose_verb_negative': 122, 'support_modality_predictive': 118, 'oppose_verb_certainty': 108, 'support_verb_doubt': 105, 'oppose_modality_necessity': 98, 'support_adj_certainty': 97, 'oppose_verb_positive': 95, 'oppose_adj_positive': 93, 'oppose_verb_doubt': 89, 'oppose_adj_certainty': 79, 'oppose_verb_hedge': 70, 'support_modality_possibility': 62, 'support_verb_con': 60, 'support_adj_pro_pattern': 57, 'support_modality_necessity': 56, 'oppose_modality_possibility': 54, 'oppose_adv_hedge': 53, 'oppose_adj_con': 39, 'support_adv_hedge': 36, 'oppose_adj_con_pattern': 34, 'support_adj_pro': 27, 'support_adj_negative': 26, 'support_verb_negative': 24, 'oppose_adj_hedge': 20, 'support_adj_hedge': 19, 'support_adv_positive': 18, 'support_adv_pro': 17, 'oppose_adj_doubt': 12, 'oppose_verb_emphatic_pattern': 12, 'oppose_adv_con': 11, 'support_verb_hedge': 10, 'oppose_adv_negative': 10, 'oppose_adv_pro': 9, 'support_adv_emphatic_pattern': 8, 'support_adj_con_pattern': 8, 'support_verb_pro_pattern': 8, 'support_adv_con': 8, 'oppose_adj_pro_pattern': 8, 'oppose_adv_doubt': 8, 'oppose_adj_pro': 8, 'support_adj_doubt': 7, 'oppose_adv_positive': 7, 'support_verb_con_pattern': 6, 'oppose_adv_con_pattern': 6, 'oppose_adv_emphatic_pattern': 6, 'support_verb_emphatic_pattern': 3, 'support_adv_negative': 3, 'support_adv_doubt': 2, 'support_verb_certainty_pattern': 1})

# Sorting the data for display
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# Start of the HTML file
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Feature Frequency Table</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h2>Linguistic Feature Frequencies</h2>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Frequency</th>
            </tr>
        </thead>
        <tbody>
"""

# Adding rows to the HTML table
for feature, freq in sorted_data.items():
    html_content += f"            <tr><td>{feature}</td><td>{freq}</td></tr>\n"

# Closing the HTML file
html_content += """
        </tbody>
    </table>
</body>
</html>
"""

print(html_content)
