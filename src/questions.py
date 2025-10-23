import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os


def extract_indiabix_questions(url):
    """Extract questions, options, and answers from one IndiaBix page"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    questions_data = []
    question_containers = soup.find_all("div", class_="bix-div-container")

    for div in question_containers:
        # Question text
        question_tag = div.find("div", class_="bix-td-qtxt")
        question = question_tag.get_text(" ", strip=True) if question_tag else "NA"

        if question == "NA":
            continue

        # Options
        options = []
        option_tags = div.find_all("td", class_="bix-td-option")
        for opt in option_tags:
            text = opt.text.strip()
            if text and not re.match(r'^[A-D]\.?$', text):
                options.append(text)

        if not options:
            option_divs = div.find_all("div", class_="bix-td-option-val")
            for opt in option_divs:
                text = opt.text.strip()
                if text:
                    options.append(text)

        if not options:
            option_table = div.find("table", class_="bix-tbl-options")
            if option_table:
                rows = option_table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        option_text = cells[1].get_text(strip=True)
                        if option_text:
                            options.append(option_text)

        while len(options) < 4:
            options.append("NA")
        options = options[:4]

        # Correct answer
        answer = "NA"
        ans_span = div.find("input", {"class": "jq-hdnakq"})
        if ans_span:
            answer_value = ans_span.get("value", "").strip()
            try:
                if answer_value.isdigit():
                    answer_index = int(answer_value) - 1
                elif answer_value.upper() in ['A', 'B', 'C', 'D']:
                    answer_index = ord(answer_value.upper()) - ord('A')
                else:
                    answer = answer_value
                    raise ValueError("Unknown answer format")

                if 0 <= answer_index < len(options) and options[answer_index] != "NA":
                    answer = options[answer_index]
                else:
                    answer = answer_value
            except:
                answer = answer_value if answer_value else "NA"

        questions_data.append({
            "Question": question,
            "OptionA": options[0],
            "OptionB": options[1],
            "OptionC": options[2],
            "OptionD": options[3],
            "Answer": answer
        })

    return questions_data


def append_page_to_csv(page_url, csv_filename):
    """Extract questions from a single page and append to an existing CSV file"""

    print(f"🔍 Extracting from: {page_url}")
    questions = extract_indiabix_questions(page_url)

    if not questions:
        print("🚫 No questions found on this page.")
        return

    new_df = pd.DataFrame(questions)

    if os.path.exists(csv_filename):
        # Append without writing header
        new_df.to_csv(csv_filename, mode='a', index=False, header=False, encoding="utf-8")
        print(f"✅ Appended {len(new_df)} questions to {csv_filename}")
    else:
        # Create new file if it doesn't exist
        new_df.to_csv(csv_filename, index=False, encoding="utf-8")
        print(f"✅ Created new CSV and added {len(new_df)} questions")


if __name__ == "__main__":
    # === EXAMPLE USAGE ===
    # Existing CSV filename
    csv_file = "indiabix_profit_and_loss_questions.csv"

    # New page URL you want to scrape and append
    page_url = "https://www.indiabix.com/aptitude/time-and-distance/037001"

    append_page_to_csv(page_url, csv_file)
