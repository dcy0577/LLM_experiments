import os
import re
from textwrap import fill
from bs4 import BeautifulSoup
from typing import List
from markdownify import MarkdownConverter


class CustomConverter(MarkdownConverter):
    def convert_p(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text
        if self.options['wrap']:
            text = fill(text,
                        width=self.options['wrap_width'],
                        break_long_words=False,
                        break_on_hyphens=False)
        return '%s\n' % text if text else ''

# Create shorthand method for conversion
def md_custom(html, **options):
    return CustomConverter(**options).convert(html)


def remove_img_tags(data):
    p = re.compile(r'<img.*?/>')
    return p.sub('', data)

def clean_html(input_path_list: List[str], output_folder_path: str):
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    cleaned_html_list = []

    for input_path in input_path_list:
        with open(input_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
            # remove xml declaration
            html_string=re.sub("\\<\\?xml(.+?)\\?\\>", "", html_string)
            
            cleaned_html_path = os.path.join(output_folder_path, os.path.basename(input_path))
            with open(cleaned_html_path, 'w', encoding='utf-8') as f:
                # clean html
                soup = BeautifulSoup(html_string, 'lxml')
                # remove title
                for m in soup.find_all('title'):
                    m.extract()
                # remove hyperlinks
                for m in soup.find_all('a'):
                    m.replaceWithChildren()
                # remove images
                for m in soup.find_all('img'):
                    m.replaceWithChildren()
                for m in soup.find_all('p'):
                    # if m.text == None or m.text == ' ' or m.text == '':
                    #     m.replaceWithChildren()
                    # remove related topics links
                    if m.has_attr('class'):
                        if 'RelatedTopics' in m['class']:
                            m.extract()
                # reomve script
                for m in soup.find_all('script'):
                    m.extract()
                
                print(soup)
                f.write(str(soup))
                cleaned_html_list.append(cleaned_html_path)

    return cleaned_html_list


def getFileList(dir, file_list, ext="htm"):
    """
    Get the file path recursively
    """
    newDir = dir
    if os.path.isfile(dir) and ext in dir[-3:]:
        file_list.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, file_list, ext)
 
    return file_list

# convert html file to markdown format file
def html2makrdown(input_path_list: List[str], output_folder_path: str):
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    for input_path in input_path_list:
        with open(input_path, 'r', encoding='utf-8') as f:
            # remove \n between tags
            html_string = f.read().split('\n')
            html_string = ''.join(html_string)

            base = os.path.basename(input_path)
            md_name = os.path.splitext(base)[0] + '.md'
            md_path = os.path.join(output_folder_path, md_name)
            with open(md_path, 'w', encoding='utf-8') as f:
                md_string = md_custom(html_string, heading_style="ATX")
                f.write(md_string)


if __name__ == "__main__":
    # for data preprocessing, create decent md files from cleaned html files
    input_path_list = getFileList('./data', [], ext="htm")
    cleaned_html_list = clean_html(input_path_list, './data_cleaned')
    html2makrdown(cleaned_html_list, './data_markdown')


