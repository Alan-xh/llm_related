import os

path = os.path.dirname(os.path.abspath(__file__))

file_path = path + '/doc/concrete.html'


from bs4 import BeautifulSoup
from bs4.element import NavigableString
import json


def find_text_with_content_recursive(soup):
    text_with_content = []
    black_tag = ['div']

    def __recursive_find_text_with_content(element):
        if isinstance(element, NavigableString) and element == '\n':
            return
        contents = element.contents

        # 多个元素判断
        if len(contents) > 1:
            if (
                isinstance(contents[0], NavigableString)
                and contents[0] != '\n'
                and element.name not in black_tag
            ):
                text_with_content.append(element)
                return
            elif contents[0].name == 'strong':
                text_with_content.append(element)
                return
            for child in contents:
                __recursive_find_text_with_content(child)
            return

        # 单个元素判断
        if element.get_text() == '':
            return
        if isinstance(contents[0], NavigableString) and element.name not in black_tag:
            text_with_content.append(element)
        elif contents[0].name in black_tag:
            __recursive_find_text_with_content(contents[0])
        else:
            text_with_content.append(element)

    __recursive_find_text_with_content(soup)
    return text_with_content


def html_to_json_with_markdown(html_file_path):

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    text_with_content = find_text_with_content_recursive(body)

    results = []
    answer_elements = []
    flag = False
    for element in text_with_content:
        if element.name == 'h1':
            if flag:
                results.append({"query": query, "answer": "".join(answer_elements)})
            flag = True
            query = element.get_text(strip=True)
            answer_elements = []
            markdown_answer = ""

        elif element.name == 'h2':
            markdown_answer = f"## {element.get_text(strip=True)}\n"
            answer_elements.append(markdown_answer)
        elif element.name == 'li':
            markdown_answer = f"- {element.get_text(strip=True)}\n"
            answer_elements.append(markdown_answer)
        elif element.name == 'script' or element.get('id') == 'footer-text':
            continue
        else:
            markdown_answer = f"{element.get_text(strip=True)}\n"
            answer_elements.append(markdown_answer)
    results.append({"query": query, "answer": "".join(answer_elements)})
    return results


if __name__ == '__main__':
    # 替换为你的HTML文件路径
    html_file = file_path
    json_data = html_to_json_with_markdown(html_file)
    print(json.dumps(json_data, indent=4, ensure_ascii=False))

    # 可选：将JSON数据保存到文件
    with open('./MEPER场景.jsonl', 'a+', encoding='utf-8') as outfile:
        for line in json_data:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')
