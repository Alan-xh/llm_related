import os
import pathlib
import base64
import logging
from openai import OpenAI
import concurrent

from typing import List, Optional, Tuple, Dict

dic_path = os.path.dirname(os.path.abspath(__file__))

imgs_path = dic_path + '/output'


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_img(doc):
    pass


DEFAULT_PROMPT = """使用markdown语法，将图片上的图片信息转成说明文档。你必须做到：
1. 对于图片部分（一般都用颜色的矩阵和png标注出来了），则根据图片内容输出使用说明文档。
2. 对于文字部分则直接输出文字。
2. 输出和使用识别到的图片的相同的语言。
3. 忽略掉长直线、忽略掉页码。"""

DEFAULT_RECT_PROMPT = "图片中用带颜色的矩形框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。"

DEFAULT_ROLE_PROMPT = """"你是一个说明文档生成助手，为软件概念图生成详细的说明文档。"""


def openai_parse_images(
    api_key: str,
    base_url: str,
    llm: str,
    image_infos: List[Tuple[str, List[str]]],
    prompt_dict: Optional[Dict] = None,
    output_dir: str = './',
    gpt_worker: int = 5,
    **args,
) -> str:
    """
    Parse images to markdown content by openai api.
    """

    if isinstance(prompt_dict, dict) and 'prompt' in prompt_dict:
        prompt = prompt_dict['prompt']
        logging.info("prompt is provided, using user prompt.")
    else:
        prompt = DEFAULT_PROMPT
        logging.info("prompt is not provided, using default prompt.")
    if isinstance(prompt_dict, dict) and 'rect_prompt' in prompt_dict:
        rect_prompt = prompt_dict['rect_prompt']
        logging.info("rect_prompt is provided, using user prompt.")
    else:
        rect_prompt = DEFAULT_RECT_PROMPT
        logging.info("rect_prompt is not provided, using default prompt.")
    if isinstance(prompt_dict, dict) and 'role_prompt' in prompt_dict:
        role_prompt = prompt_dict['role_prompt']
        logging.info("role_prompt is provided, using user prompt.")
    else:
        role_prompt = DEFAULT_ROLE_PROMPT
        logging.info("role_prompt is not provided, using default prompt.")

    def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
        logging.info(f'gpt parse page: {index}')

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # agent = Agent(role=role_prompt, api_key=api_key, base_url=base_url, disable_python_run=True, model=model, **args)
        page_image, rect_images = image_info
        local_prompt = prompt
        local_prompt = role_prompt + local_prompt

        """显示图片"""
        # if rect_images:
        #     local_prompt = local_prompt + (rect_prompt % ', '.join(rect_images))

        # content = agent.run([local_prompt, {'image': page_image}], display=verbose)
        page_image_b64 = encode_image(page_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{page_image_b64}",
                        },
                    },
                    {"type": "text", "text": local_prompt},
                ],
            }
        ]

        completion = client.chat.completions.create(model=llm, messages=messages)

        output_text = completion.choices[0].message.content

        return index, output_text

    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = [
            executor.submit(_process_page, index, image_info)
            for index, image_info in enumerate(image_infos)
        ]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            content = content
            print(content)

            # 在某些情况下大模型还是会输出 ```markdown ```字符串
            if '```markdown' in content:
                content = content.replace('```markdown\n', '')
                last_backticks_pos = content.rfind('```')
                if last_backticks_pos != -1:
                    content = (
                        content[:last_backticks_pos] + content[last_backticks_pos + 3 :]
                    )

            contents[index] = content

    output_path = os.path.join(output_dir, '工具手册.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(contents)):
            f.write(f"\n\n第{i+1}页\n\n")
            f.write(contents[i])

    return '\n\n'.join(contents)


if __name__ == '__main__':
    image_infos = []
    image_info = ()
    for index in range(197):
        image_info = (os.path.join(imgs_path, f'{index}.png'), [])
        image_infos.append(image_info)

        for img in os.listdir(imgs_path):
            if img.split('_')[0] == str(index):
                image_info[1].append(img)

    openai_parse_images(
        api_key="8ca11906-427f-40de-b326-cc9c7fcb9913",
        base_url="https://ark.cn-beijing.volces.com/api/v3/",
        llm="doubao-1.5-vision-pro-32k-250115",
        image_infos=image_infos,
        output_dir='./',
        gpt_worker=10,
    )
