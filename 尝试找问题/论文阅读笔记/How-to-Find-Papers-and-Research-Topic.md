# How to find papers and research topics in CS

> https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/
>
> https://ying-zhang.cn/misc/2016-we-love-paper/
>
> 

Minel Huang

Last updated on Oct 10, 2021 2 min read

#### ⚠ 转载请注明出处：*协作者：MinelHuang，更新日期：June.15 2021*

[![知识共享许可协议](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-nd/4.0/)

  本作品由 **MinelHuang** 采用 [知识共享署名-非商业性使用-禁止演绎 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-nd/4.0/) 进行许可，在进行使用或分享前请查看权限要求。若发现侵权行为，会采取法律手段维护作者正当合法权益，谢谢配合。

## 目录

您可以通过目录直接阅读您感兴趣的部分，各章节间并无太大联系。

  Section 0. [**前言**](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/#0_preface)：介绍了本章所阐述的内容

  Section 1. [**文献类型**](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/#section1)：介绍笔者所接触到的文献类型与分级

  Section 2. [**文献检索**](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/#section2)：介绍几种检索文献的网站与文献质量评估方法

  Section 3. [**寻找方向**](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/#section3)：描述笔者寻找研究方向的方法和评判标准



## 0. 前言

  本文意在描述在科研的起步阶段（时间点为研究生/博士生入学前）如何寻找个人的研究方向，并提高该研究方向可以产出结果的概率。在此本文将简述三个方面的必备知识，包括：

1. **文献分类**：描述文献有哪几部分组成，哪些类型的文献是优质的，哪些类型的文献是在起步阶段对个人最有帮助的。
2. **文献检索**：对于所需的文献如何检索，如何评估一篇文献的质量是高的。
3. **寻找方向**：总结笔者寻找research topic的一般方法。

  本文当前是笔者在入学前根据周围师兄/导师以及网上的研究方法进行的总结，您在参考本文提出的方法时请结合自身情况再做具体判断。探索适合自己的科研方法是最重要的，**常借鉴，常思考，常总结**。

## 1. 文献类型

参考文章链接：

1. [如何收集和整理论文（面向CS专业）](https://ying-zhang.github.io/misc/2016-we-love-paper/) 作者：Ying ZHANG

  通常将文献分为期刊与会议论文，其整体质量为期刊>会议。但对于CS领域而言，由于期刊的审稿与修改时间较为漫长，通常时间跨度在一年左右，并且期刊通常按季度发布（每期称为Issue）、每期收录论文较少，这都不太符合CS领域的高速发展现状。故在CS领域中，大多数成果都将在会议论文中，期刊主要收录一些理论性较强或综述类论文，这是与传统学科有所区别的地方。故作为CS学生，接触的较多论文为**会议论文**。

#### 具体文献分类

  期刊分为**Transaction, Journal和Magazine**，三者学术严肃性依次降低。严格来说Magazine不算是学术期刊了，上面很少发表新的原创性的内容，而是对当前进展的简介和综述，也会转发一些已经发表过的重要的Paper。对于新手而言，了解一个Topic的方法可以是通过Magazine中的**综述类论文**，建立一些基本的概念以及找到该领域中的关键词，这会对后面的研究有相当大的帮助。

  会议论文是在周期型举办的学术会议上发表的论文，一般而言每次学术会议举办前都会给出一个call for paper（cpf，征稿启事），并且提供征稿的截止日期（ddl）。世界各地的学者需在ddl前向会议投稿，经审稿、修改后将被会议收录，而作者最后需在会议举办时赴会做Presentation。
  会议中被录用的文章会结集出版，称为**Proceedings**，有些会议还会对论文进行评奖（杰出论文等）。一些优秀的论文可能会被推荐到合作的期刊，由作者进行扩展最后作为期刊论文发表。
  会议分为**Symposium，Conference和Workshop**，其学术严肃性依次降低，大部分会议都被称为Conference。这里对会议的分级参考**CCF**（中国计算机学会）目录，其将会议/期刊为A、B、C三类，具体可参考[中国计算机学会推荐国际学术会议和期刊目录-2019](https://www.ccf.org.cn/Academic_Evaluation/By_category/)。

相信大家更多接触到的文献描述词汇为**SCI和EI**，其网址如下：

- EI **Engineering Index** https://www.engineeringvillage.com/
- SCI **Science Citation Index** https://apps.webofknowledge.com/

  看名称也可以发现，SCI与EI仅是针对文章的收录（包括所有领域的论文），上述两个数据库的用途基本上**仅为已知文章标题，检索是否被收录**，而**不是**真正用于收集文章。

  根据以上分类，笔者自身会先根据具体会议方向描述和CCF评级确定是否在此会议中检索，而后通过标题与摘要迅速获取论文信息（扫论文），确定接下来的一周内将阅读哪些论文全文。

#### 推荐会议集

- [**ACM SIGs** ](https://dl.acm.org/sigs)：
  ACM之下针对CS多个子方向的“分舵”，称为Special Interest Group, SIG。其子方向分的较为详细，一些会议网站中还维护了部分会议的最佳论文列表以及后文将介绍的USENIX的各会议最佳论文。
- [**USENIX** ](https://www.usenix.org/conferences)：
  USENIX最初称为Unix User Group。它组织了OSDI 、ATC、FAST、NSDI、LISA等会议，不但学术水平很高，贴近工业界，而且免费提供全文下载，还提供一些论文作者在会议上的slides及演讲视频。

  下面给出一些系统方向的重要会议：

1. SOSP：ACM出版社（CCF分类：软件工程/系统软件/程序设计语言）
2. OSDI：USENIX出版社（CCF分类：软件工程/系统软件/程序设计语言）
3. SIGCOMM：ACM出版社（CCF分类：计算机网络）
4. NSDI：USENIX出版社（CCF分类：计算机网络）

  可以通过上述4类会议来了解当前计算机领域中的热点研究方向（系统方面，不包含AI），在第三章中将描述如何运用会议论文寻找与确定研究方向。

## 2. 文献检索

参考文章链接：

1. [如何收集和整理论文（面向CS专业）](https://ying-zhang.github.io/misc/2016-we-love-paper/) 作者：Ying ZHANG

  一般会议和期刊都有自己的网站，但很少能在上面获取全文。又因为来源分散，直接从它们那里检索Paper很不方便。有几个大型的论文数据库，它们与期刊或者会议主办方合作，或者自己组织会议或编辑出版期刊，比如下面的表格：

| 机构                                     | Digital Library(DL) | 链接                                                         |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------ |
| Association for Computing Machinery, ACM | ACM Digital Library | [dl.acm.org](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/dl.acm.org) |
| IEEE Computer Society                    | IEEE Xplore DL      | [ieeexplore.ieee.org/](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/ieeexplore.ieee.org/) |
| Elsevier ScienceDirect                   | -                   | [www.sciencedirect.com](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/www.sciencedirect.com) |
| Springer                                 | Springer Link       | [www.springer.com](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/www.springer.com) |
| Wiley                                    | Wiley Online Lib    | [onlinelibrary.wiley.com](https://huangxyminel.netlify.app/post/21-06-15-how_to_find_papers_and_find_research_topic_in_cs/onlinelibrary.wiley.com) |

  ACM 和 IEEE Computer Society（计算机学会，IEEE还有电气、电子、通信等其它多个学会） 的网址后缀是 .org，这两个是CS领域最重要的学术组织，很多的CS学术会议都是由它们组织的。 Elsevier，Springer，Wiley的网址后缀则是 .com ，这些是学术出版商，内容以期刊为主，涵盖了CS及其它多个学科。 上面这几个数据库是 主要的论文全文来源。它们各自收录的会议和期刊基本没有重叠，从它们的数据库下载的Paper也都有各自的排版样式。

  ACM作为最“正统”的计算机学术组织，它的DL除了收录ACM组织的会议和期刊全文之外，还会索引其它几家数据库的 **元数据**，但没有全文，可以通过DOI链接跳转到这几家数据库的全文页面。
  IEEE出版的一些论文在 computer.org （实际是CSDL: [www.computer.org/csdl/](https://www.computer.org/csdl/)）和 Xplore DL 都可能搜到，这两个数据库是 **分别** 收费的，能在Xplore DL下载的不一定能在Computer.org下载。

  除数据库外，笔者在检索论文中遇到的困难大多是没有正确的关键词、找到的论文和个人方向不符、文章看完后云里雾里的。这些问题笔者的解决方法如下：

1. **选择合适的数据库检索**：在第一章中本文提到了CS领域各个方向的会议，一个显然的办法是以该数据集为约束进行检索。通常使用Google Scholar进行搜索，该搜索引擎的匹配功能非常好，可以在高级搜索选项中添加各个约束。如若搜不到再考虑使用ACM，IEEE Explore自带的搜索引擎，几乎不使用Web of Science进行检索（范围太广很难筛选到自身方向的论文）。
2. **查找关键词**：关键词实际上应该在寻找方向/idea阶段得到，在该阶段，笔者的检索方式为扫会议论文摘要，选择感兴趣的全文阅读，而后在论文中的Problem部分提取关键词。通过Wiki、Google Scholar等方式拓展该关键词（实际上是对Problem的补充）。
3. **有了研究方向后该如何检索**：一个很好的办法是对已经提取出的文章的参考文献进行溯源，以及继续关注该文献被谁引用。参考文献的溯源较为简单，因为已经找到了文章标题，检索即可。关于被引，推荐谷歌学术的引用快讯（科研人员人工添加的引用链接）和作者快讯（可以获取该作者的新文章、新引用）。

## 3. 寻找方向

  研究方向是一份科研工作的起始，“好”的方向可以说是决定了产出的频率和质量，在此笔者希望总结自己对方向的看法以及一套方法论，该看法仅为笔者先阶段的思想，随着不断学习将逐步修改。

  第一个问题，到底怎样才能算是一个研究方向？在笔者最初的思想中，方向是一个指导性的、面向一个领域的很宽泛的东西。比如，笔者最开始以为自身的方向为“分布式计算”，并以分布式计算为关键词进行了多种方式的检索。实际上，“分布式计算”仅仅能作为一个领域的概述，而无法作为一个Topic，更不能作为一个Idea。实际上，笔者认为找方向就是在对领域做不断细化，直至细化到一个Problem与几个Solution，当我们可以针对这个Problem提出几个可能的Solution后，便可以算真正“找到方向了”。
  当然有的朋友会想，如果我都想出Solution了，整个论文不就做完了吗，还用找什么方向？在这里需要对真正的Solution进行区分，我们所想的Idea，是在**完全理解**Problem的基础上，对**前人的Solution进行分类**后，自己产生的几个“脑洞”。在这一阶段我们不需要整明“脑洞”是正确的，也未曾想过是否有意义或者到底有没有优化甚至不考虑能不能做出来，仅仅需要自己能产出那么几个零星的想法即可。后文将介绍笔者自己的找方向方法论，在叙述方法时您应该可以体会到笔者对“研究方向与Idea”的理解。

#### Step 1：选择一个Topic

  CCF或各大会议的分类实际上已经对计算机领域做了划分，但显然这些领域还是过于宽泛。一个Topic是一组问题的集合，比如System for DL（为深度学习设计的系统），寻找Topic的关键在于是否能找到一个场景。在笔者初入科研时，常常会想做出一个“大一统”的系统，即一个系统完成所有功能，不涉及场景，实际上就是没有细化领域。每一个one-for-all的文章都是极有贡献度的，但并不适合笔者这种刚进入科研的萌新。大牛是在基于对该领域的全面理解上做总结，从而产生的one-for-all式的研究，上手时依然是选择一个小领域深究。**入门切勿只追求广度而不追求深度**，而对一个领域进行了初部的划分的标志便是，**找到了一个场景**。笔者总结的方法论如下：

1. **选择一个感兴趣的领域**：参考CCF和会议分类，通过Wiki、Youtube等方式对该领域快速了解。

2. **划分领域与提取关键词**：通过阅读会议论文中的Magazine（十分推荐ACM中的[Communications of the ACM, CACM](https://dl.acm.org/magazine/cacm)和[Queue](https://dl.acm.org/magazine/queue)），通过阅读标题和摘要，对现有研究进行一个粗略的问题分类（比如调度问题、负载均衡问题、效率问题、节能问题等等），选择几个自己感兴趣的问题，提取出一些关键词。

3. **概况Topic**：在A类推荐会议列表中（如第一章提到的SOSP，OSDI等）使用2中的关键词进行检索，查看提取出的分类是否为研究热点，而后重点关注每个问题所检索出的论文是属于何种场景的。一般我们可以用一组短语概括，如Schedule for DNN。于是我们得到了Problem for Scenario的二元组，每个二元组可以概括成一个Topic。

4. **细化Problem**：根据Topic，已经可以很准确的提取近几年的热点论文了，几乎每个热点论文都会对3中的Problem进行细化，比如Schedule for DNN中的二层交换机由于DNN的某一特性发生拥塞问题。寻找多个感兴趣的诸如此类的Problems是此步的目的。当然此时看论文还可以借助Review的帮助（这都是基于Topic的关键词），在Review中会对Problem进行大量的细化与分类。

  经历上述几步，笔者便可以找到一个自己感兴趣的Topic，并了解了该Topic下的近年来的热点Problem和热点论文。在此，笔者才能确认自身确实对该Topic感兴趣并有一定的了解。

#### Step 2：总结前人的Solution

  在确定了Topic后，下一步需要发现新的Problem。但对于一个小白而言，Problem不可能是凭空出现的，也很难是灵光乍现得出的。所以在这里笔者认为，总结与复现前人的工作是很有必要的。这里需要确保自己选择的Problem是有意义的，切勿对一个毫无意义的Problem继续研究，或者是该类Problem在近五年没有前人继续研究（说明很可能这个Problem被做完了）。
   所以该步重点在于精确检索，总结前人的研究方法和结果，需要精读热点论文，而非Step 1中的略读、泛读论文。实际上笔者喜欢将前人的工作比作“轮子”，我们的工作是在旧轮子的基础上造新轮子，或者优化旧轮子，而不是凭空造轮子或者重复造轮子。所以不是要百分百理解前人的Solution，而是明白其方法论是什么，不需要硬琢磨其中的数学推导。当然，可能需要借助复现的手段辅助理解论文，并不是在有了Idea后再动手，笔者认为动手做依然是关键的一步。

#### Step 3：产生Idea

  精读与复现前人工作时很有可能觉得换种方法可能更好，这里笔者的经验是，在编程复现时有一些点会发现论文叙述的很笼统，或者没有考虑到，或者某一部分的流程看着不顺眼。这便可以称作是Idea了，那这些Idea难道是做完了吗？显然是不的，不论是修改部分方法，还是更换方法都可能涉及其他的问题，这里的“其他问题”便是我们以后的所发布论文的研究点。

#### Step 4：攥写Research Proposal

  寻找方向的最后一步便是产出RP，或者说每一步都对应RP中的一部分，比如Step 1对应Introduction，Step 2对应Related work，Step 3对应新的想法。在RP中需要对上述三步进行一遍详细的梳理，确保由背景产生动机，由动机产生问题，由问题产生解决方案这一思路是顺畅的。最后需要给出一个可行的研究计划与方案。

#### Step 5：开始研究

  于此，我们便可以根据RP，一步一步的开始我们的研究。比如包括对问题的数学建模、设计优化算法、搭建实验平台与验证等等，而后的研究才算有了方向。所以，笔者认为寻找方向的最后标志便是产出RP。