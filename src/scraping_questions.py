import pandas as pd
import mysql.connector
from mysql.connector import Error

# ====== MySQL Configuration ======
db_config = {
    'host': 'localhost',
    'user': 'root',       # replace with your MySQL username
    'password': 'dhivya@2005',       # replace with your MySQL password
    'database': 'mockmate_db'  # replace with your database name
}

csv_file = r"E:\Codings\Project_work_I\indiabix_verbal_questionss.csv"
category_name = 'Verbal'

def insert_questions_from_csv(csv_file, category):
    try:
        # Read CSV and fill NaN with empty string
        df = pd.read_csv(csv_file).fillna('')

        # Connect to MySQL
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        for index, row in df.iterrows():
            sql = """
                INSERT INTO aptitude_questions
                (category, question, option_a, option_b, option_c, option_d, correct_answer)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            # Map answer text to A/B/C/D
            correct_answer = 'NA'
            options = [row['OptionA'], row['OptionB'], row['OptionC'], row['OptionD']]
            answer_text = str(row['Answer']).strip()
            if answer_text in options:
                correct_answer = ['A','B','C','D'][options.index(answer_text)]

            values = (
                category,
                row['Question'],
                row['OptionA'],
                row['OptionB'],
                row['OptionC'],
                row['OptionD'],
                correct_answer
            )
            cursor.execute(sql, values)

        connection.commit()
        print(f"✅ Inserted {len(df)} questions into the '{category}' category.")

    except Error as e:
        print(f"❌ Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    insert_questions_from_csv(csv_file, category_name)
