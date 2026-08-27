from __future__ import annotations

import json

PINYIN = """
shishi gongsi fazhan yanjiu jishu shuju moxing xitong gongcheng kaifa
ceshi fenxi wenti jieguo fangfa jizhi sheji xingneng anquan yinsi
shijie zhongguo beijing shanghai xuexi xuesheng laoshi daxue yuyan wenhua
lishi jingji shehui zhengce guanli fuwu chanpin yonghu shichang qiye
xiangmu jihua mubiao renwu shijian kongjian huanjing ziyuan nengyuan xinxi
wangluo jisuan ruanjian yingjian shouji diannao pingtai yingyong gongju wenjian
baogao huiyi tongzhi xiaoxi xinwen neirong wenzhang zuozhe duzhe chengshi
nongcun jiaotong yisheng yiyuan jiankang jiaoyu kexue ziran renlei guojia
zhengfu falv guize biaozhun zhiliang xiaolv sudu yanchi jiyi cunchu
xunlian pinggu zhunquelv houxuan pinyin shuru chongpai shangxiawen xinxin baoshou
""".split()

CONTEXTS = (
    "根据公开资料显示",
    "这个项目需要",
    "研究人员正在",
    "我们将继续",
    "报告重点讨论",
)


def main() -> None:
    if len(PINYIN) != 100 or len(set(PINYIN)) != 100:
        raise RuntimeError("fixture request list must contain exactly 100 unique pinyin strings")
    for index, pinyin in enumerate(PINYIN):
        print(json.dumps({"pinyin": pinyin, "context": CONTEXTS[index % len(CONTEXTS)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
