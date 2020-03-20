import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os

num = 0
numPicture = 0
file = ''
List = []


def Find(url):
    global List
    print('INFO:detecting all images,please waiting.....')
    t = 0
    i = 1
    s = 0
    while t < 1000:
        Url = url + str(t)
        try:
            Result = requests.get(Url, timeout=7)
        except BaseException:
            t = t + 60
            continue
        else:
            result = Result.text
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片url
            s += len(pic_url)
            if len(pic_url) == 0:
                break
            else:
                List.append(pic_url)
                t = t + 60
    return s


def recommend(url):
    Re = []
    try:
        html = requests.get(url)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re


def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    print('INFO:find keyword:' + keyword + '\'s image，downloading...')
    for each in pic_url:
        print('INFO:downloading:' + str(num + 1) + ' images，image address:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('Error，can not download this image')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return

if __name__ == '__main__':  # 主函数入口
    tm = int(input('Please input the num of each name:'))
    numPicture = tm
    line_list = []
    with open('./name.txt', encoding='utf-8') as file:
        line_list = [k.strip() for k in file.readlines()]  # 用 strip()移除末尾的空格
    for word in line_list:
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
        tot = Find(url)
        Recommend = recommend(url)  # 记录相关推荐
        print('INFO:through detection %s has %d images' % (word, tot))
        file = word + '文件'
        y = os.path.exists(file)
        if y == 1:
            print('INFO:file exists,input again')
            file = word+'2'
            os.mkdir(file)
        else:
            os.mkdir(file)
        t = 0
        tmp = url
        while t < numPicture:
            try:
                url = tmp + str(t)
                result = requests.get(url, timeout=10)
                print(url)
            except error.HTTPError as e:
                print('INFO:timeout error，adjust the network')
                t = t + 60
            else:
                dowmloadPicture(result.text, word)
                t = t + 60
        numPicture = numPicture + tm
    print('INFO:download finished')
