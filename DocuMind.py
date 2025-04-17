# CLAP: Chat Local And Persistent，网络隐私敏感场景下语境对话可加载设计实现的基于Ollama框架的本地大语言模型语义互动软件

###### 版本号：1.0.0

"""
Based on Ollama, a Graphical User Interface for Loc al Large Language Model Conversations.
"""
#权限检查
import os
os.environ["MPLBACKEND"] = "module://matplotlib.backends.backend_qtagg"
try:
    with open("test_write.txt", "w") as f:
        f.write("test")
    os.remove("test_write.txt")
except Exception as e:
    print(f"文件权限异常: {str(e)}")
    exit(1)
from os.path import exists
import time
import base64
import importlib.metadata
import shutil
import stat
import sys
import hashlib
import json
import pickle
import sqlite3
import re
import tempfile
import random
from matplotlib.colors import to_rgba
from docx import Document
import webbrowser
import traceback
from nltk.tokenize import word_tokenize
from collections import Counter
import community.community_louvain as community_louvain
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.path as mpath
from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
from matplotlib.animation import FuncAnimation
from matplotlib.patheffects import withStroke
from itertools import cycle
from PySide6.QtWidgets import QTabWidget, QGraphicsOpacityEffect, QDialog,QDialogButtonBox, QFileDialog, QProgressDialog, QApplication, QMainWindow, QWidget, QToolBar, QTextEdit, QListWidget, QListWidgetItem, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextBrowser, QFileDialog, QMessageBox
from PySide6.QtCore import QDateTime, QTimer, QObject, QEasingCurve, QPropertyAnimation, Slot, QMetaObject, QMutex, QWaitCondition, QMutexLocker, Qt, QUrl, QThread, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtMultimedia import QSoundEffect
import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.config import Settings
import numpy as np
import itertools
import math
import ollama
import json
import pickle
import sqlite3
import sys
import itertools
import math
import adjustText
import spacy
import jieba
from jieba.posseg import pair
import jieba.posseg as pseg
from spacy import displacy
from spacy.tokens import Span
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import mammoth
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from ltp import LTP
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections
from matplotlib.collections import LineCollection
from matplotlib import rcParams
from PIL import Image

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

# 字体自动检测
try:
    from matplotlib.font_manager import findfont, FontProperties
    zh_font = findfont(FontProperties(family=['SimHei', 'Microsoft YaHei', 'sans-serif']))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
except:
    print("中文字体未找到，将使用默认字体")
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from datetime import datetime
from PySide6.QtGui import QColor, QTextDocument, QTextDocumentFragment, QAction, QFont, QGuiApplication, QKeySequence,QShortcut, QTextCursor, QDragEnterEvent, QDropEvent, QMovie
from PySide6.QtWidgets import QFrame, QComboBox,QAbstractItemView, QHBoxLayout, QLabel, QMainWindow, QApplication, QMenu, QSizePolicy, QSplitter, QTextBrowser, QTextEdit, QWidget, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy
from PySide6.QtCore import Q_ARG, QAbstractTableModel, QModelIndex, QUrl, QVariantAnimation, Qt, QTranslator, QLocale, QLibraryInfo, QThread, Signal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

from scipy.stats import gmean
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
# from IPython.display import display as Markdown
from tqdm.autonotebook import tqdm as notebook_tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_path)
print(current_directory)
# 改变当前工作目录
os.chdir(current_directory)

# 统一实体映射
ENTITY_MAPPING = {
    'Nh': '人物',
    'Ni': '机构',
    'Ns': '地点',
    'NORP': '民族或宗教团体',
    'PRODUCT': '产品',
    'LOC': '位置',
    'MISC': '其他实体',
    'ORG': '组织',
    'PERSON': '人名',
    'GPE': '国家或地区',
    'DATE': '日期',
    'TIME': '时间',
    'PERCENT': '百分比',
    'MONEY': '货币金额',
    'QUANTITY': '数量',
    'CARDINAL': '基数',
    'EVENT': '事件',
    'WORK_OF_ART': '作品',
    'RESEARCH_TOPIC': '研究主题',
    'JOURNAL_NAME': '期刊名称',
    'CONFERENCE_NAME': '会议名称',
    'FORMULA_SYMBOL': '公式符号',
    'SCIENTIFIC_TERM': '科学术语',
    '固有名詞': '固有名詞',
    '人物': '人物',
    '組織': '組織',
    '地域': '地域',
    '一般名詞': '一般名詞'
}

def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False
    
def post_process_html(html):
        """后处理HTML内容"""
        # 移除图片固定尺寸
        html = re.sub(r'(<img[^>]+?)width="\d+"', r'\1', html, flags=re.IGNORECASE)
        html = re.sub(r'(<img[^>]+?)height="\d+"', r'\1', html, flags=re.IGNORECASE)
        # 修复列表缩进
        html = html.replace("<li>", "<li style='margin: 5px 0'>")
        return html
    
class TerminalStyleBrowser(QTextBrowser):
    def __init__(self):
        super().__init__()
        # 启用平滑滚动
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.verticalScrollBar().setSingleStep(20)
        
    def append(self, html):
        # 保持追加时的滚动逻辑
        max_scroll = self.verticalScrollBar().maximum()
        super().append(html)
        QTimer.singleShot(10, lambda: 
            self.verticalScrollBar().setValue(max_scroll + 100)
        )


class DocumentManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.embedding_model = "deepseek-r1:8b"  # 明确指定嵌入模型
        self.expected_dim = 4096  # 根据模型实际维度设置
        self.single_docs = {}  # 独立文档存储 {路径: 向量库}
        self.collection_db = None  # 多文档集合存储
        self.current_mode = "single"  # 默认单文档模式
        self.current_collection = None
        self.progress_callback = None
        self.load_existing_db()
        self.documents = {}
        self.loaded_paths = []  # 跟踪所有加载路径
        self.conn = sqlite3.connect('qa_history.db')# 初始化数据库连接
        self.conversation_context = {}
        self.raw_texts = {}  # 用于存储原始文本
        self._create_table()
        self.load_success = False
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.save_contexts)
        self.autosave_timer.start(300000)  # 每5分钟保存一次
        # 初始化前强制清理旧库
        self.purge_old_databases()
        self.expected_dim = self._detect_embedding_dim()

    def _detect_embedding_dim(self):
        """通过样本推理自动检测嵌入维度"""
        sample_text = "dimension test"
        embeddings = OllamaEmbeddings(model="deepseek-r1:8b").embed_documents([sample_text])
        return len(embeddings[0]) if embeddings else 4096  # 默认值
    
    def _create_table(self):
        cursor = self.conn.cursor()
        # 问答历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
        # 会话上下文表  
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_context (
                session_id TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
        self.conn.commit()

    def save_contexts(self):
        try:
            cursor = self.conn.cursor()
            for session_id, context in self.conversation_context.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO conversation_context 
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (session_id, json.dumps(context)))
            self.conn.commit()
        except Exception as e:
            print(f"自动保存上下文失败: {str(e)}")
        
    def get_questions_answers(self, file_path):
        """从数据库获取问答历史"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT question, answer 
            FROM qa_history 
            WHERE file_path = ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (file_path,))
        return [
            {'question': row[0], 'answer': row[1]} 
            for row in cursor.fetchall()
        ]

    def set_questions_answers(self, file_path, qa_list):
        """批量保存问答到数据库"""
        cursor = self.conn.cursor()
        # 先清空旧记录
        cursor.execute('DELETE FROM qa_history WHERE file_path = ?', (file_path,))
        # 插入新记录
        for qa in qa_list:
            cursor.execute('''
                INSERT INTO qa_history (file_path, question, answer)
                VALUES (?, ?, ?)
            ''', (file_path, qa['question'], qa['answer']))
        self.conn.commit()

    def get_conversation_context(self, session_id, max_length=2000):
        """获取并截断对话上下文"""
        context = self.conversation_context.get(session_id, [])
        total_len = sum(len(msg) for msg in context)
        
        # 截断策略：保留最近的对话
        while total_len > max_length and len(context) > 1:
            removed = context.pop(0)
            total_len -= len(removed)
        return "\n".join(context)
    
    def update_conversation_context(self, session_id, question, answer):
        """更新对话上下文"""
        entry = f"用户问：{question}\n系统答：{answer}"
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = []
        self.conversation_context[session_id].append(entry)

    def update_doc_count(self):
        count = len(self.loaded_paths)
        self.main_window.doc_count_label.setText(f"已加载文档: {count}")

    def set_mode(self, file_count):
        self.current_mode = "collection" if file_count > 1 else "single"
        self._update_mode_label()

    def _normalize_collection_name(self, name):
       # 替换非法字符为下划线
        normalized = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
        # 替换连续下划线
        normalized = re.sub(r'_+', '_', normalized)
        # 去除首尾特殊字符
        normalized = normalized.strip('-_')
        
        # 确保至少有3个字符并且以字母数字字符开头和结尾
        if len(normalized) < 3:
            normalized = f"d_{normalized}"
        if not normalized[0].isalnum():
            normalized = f"d{normalized}"
        if not normalized[-1].isalnum():
            normalized = f"{normalized}d"
        # 截断长度至63字符
        return normalized[:63]
    
    def rebuild_collection(self, path):
        """针对单个文档的重建"""
        print(f"开始重建文档集合: {path}")
        
        try:
            # 1. 清除旧数据
            persist_dir = f"chroma_db/single/{os.path.basename(path)}"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
                
            # 2. 重新加载文档
            del self.single_docs[path]
            self._add_single_doc(path)
            
            # 3. 验证重建结果
            if path in self.single_docs and self.verify_collection():
                print(f"文档 {os.path.basename(path)} 重建成功")
                return True
                
            print(f"重建失败: {path}")
            return False
        
        except Exception as e:
            print(f"重建过程中发生错误: {str(e)}")
            return False

    def purge_old_databases(self):
        """安全清理旧数据库"""
        paths_to_check = [
            "multi_doc_db",
            "chroma_db/single", 
            "chroma_db"
        ]
        
        for path in paths_to_check:
            # 添加路径存在性检查
            if os.path.exists(path):
                try:
                    print(f"正在清理: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                except Exception as e:
                    print(f"清理失败 {path}: {str(e)}")
            else:
                print(f"目录不存在，跳过清理: {path}")  # 添加调试信息
        
    def verify_collection(self):
        """增强版集合验证"""
        try:
            if not self.current_collection:
                print("验证失败：当前集合为空")
                return False
                
            # 获取底层Chroma集合
            chroma_collection = self.current_collection._collection
            if not chroma_collection:
                print("验证失败：未获取到Chroma集合")
                return False
                
            # 基础检查
            print(f"集合状态检查:")
            print(f"- 名称: {chroma_collection.name}")
            print(f"- 文档数: {chroma_collection.count()}")
            
            # 样本数据检查
            sample = chroma_collection.peek(1)
            if not sample.get('ids'):
                print("警告：集合为空")
                return True  # 空集合仍视为有效
                
            # 元数据检查
            if 'metadatas' in sample:
                print(f"- 元数据类型: {type(sample['metadatas'][0])}")
                
            return True
            
        except Exception as e:
            print(f"集合验证异常: {str(e)}")
            traceback.print_exc()
            return False
        
    def verify_dimension(self, collection):
        """验证向量维度一致性"""
        try:
            # 获取集合中的第一个嵌入向量
            sample = collection.get(include=["embeddings"])["embeddings"][0]
            actual_dim = len(sample)
            
            if actual_dim != self.expected_dim:
                raise ValueError(
                    f"维度不匹配: 预期 {self.expected_dim} 实际 {actual_dim}"
                    "\n可能原因："
                    "\n1. 切换了不同维度的嵌入模型"
                    "\n2. 旧向量库未清理"
                    "\n解决方案："
                    "\n   a. 删除所有 chroma_db/ 和 multi_doc_db/ 目录"
                    "\n   b. 重启应用程序"
                )
            return True
        except IndexError:
            return True  # 空集合无需验证
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "严重错误",
                str(e)
            )
            self.purge_old_databases()
            return False
    
    def _update_mode_label(self):
        text = "模式：多文档联合" if self.current_mode == "collection" else "模式：独立分析"
        self.main_window.mode_label.setText(text)

    def load_existing_db(self):
        try:
            vector_db = Chroma(
                collection_name="multi_doc_rag",
                persist_directory="multi_doc_db/",
                embedding_function=OllamaEmbeddings(
                    model=self.embedding_model    
                )
            )
            self.current_collection = vector_db
        except Exception as e:
            print(f"No existing DB found or error loading DB: {e}")
            self.current_collection = None

    def add_documents(self, paths):
        try:
            # 在加载前进行维度验证
            if self.current_collection and not self.verify_dimension(self.current_collection._collection):
                return
            if not isinstance(paths, list):  # 确保输入是列表
                paths = [paths]
            new_paths = [p for p in paths if p not in self.loaded_paths]
            if not new_paths:
                return
            # 标记加载成功
            self.load_success = False

            # 定义回调函数（移动到类内部）
            def on_doc_loaded(success, loaded_path):
                if success:
                    # 使用Qt的线程安全方式调用UI更新
                    QMetaObject.invokeMethod(
                        self.main_window,  # 通过main_window访问界面组件
                        "_show_document_info",
                        Qt.QueuedConnection,
                        Q_ARG(str, loaded_path)  # 明确传递当前路径
                    )

            if self.current_mode == "collection":
                self._add_to_collection(new_paths)
                if new_paths:
                    self.load_success = True
                    on_doc_loaded(True, new_paths[0])
            else:
                for path in new_paths:
                    try:
                        self._add_single_doc(path)
                        # 单文档模式每个文件加载后立即显示
                        self.load_success = True
                        on_doc_loaded(True, path)  # 传递当前path到闭包
                    except Exception as e:
                        print(f"Error loading {path}: {str(e)}")
                        on_doc_loaded(False, path)
            
            # 更新加载路径（去重）
            new_paths = [p for p in paths if p not in self.loaded_paths]
            self.loaded_paths.extend(new_paths)
            
        except Exception as e:
            self.load_success = False
            if "dimensionality" in str(e):
                self.show_dimension_error()
            else:
                # 显示错误信息到界面
                QMetaObject.invokeMethod(
                    self.main_window.preview_info_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"⚠️ 文档加载失败: {str(e)}")
                )
                raise

    def show_dimension_error(self):
        """显示友好的维度错误提示"""
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("维度不匹配")
        msg.setText("🛑 检测到嵌入维度冲突！")
        msg.setInformativeText(
            "可能原因：\n"
            "1. 切换了不同版本的嵌入模型\n"
            "2. 残留旧版本向量库\n\n"
            "请选择处理方式："
        )
        
        # 添加操作按钮
        cleanup_btn = msg.addButton("立即清理并重启", QMessageBox.ActionRole)
        manual_btn = msg.addButton("手动清理指南", QMessageBox.HelpRole)
        cancel_btn = msg.addButton("取消", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == cleanup_btn:
            self.purge_old_databases()
            QApplication.exit(100)  # 特殊退出码触发重启
        elif msg.clickedButton() == manual_btn:
            webbrowser.open("https://github.com/yourrepo/cleanup_guide")

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def _add_single_doc(self, path):
        """独立文档存储（带进度反馈）"""
        persist_dir = f"chroma_db/single/{self._normalize_collection_name(os.path.basename(path))}"
        os.makedirs(persist_dir, exist_ok=True)

        try:
            if path in self.single_docs:
                print(f"Document already loaded: {path}")
                return

            # 通知开始加载（总步骤数）
            if self.progress_callback:
                self.progress_callback({
                    'type': 'start',
                    'filename': os.path.basename(path),
                    'total_steps': 4  # 总共有4个主要步骤
                })

            # 生成规范化的集合名称
            raw_name = os.path.splitext(os.path.basename(path))[0]
            collection_name = self._normalize_collection_name(raw_name)
            persist_dir = f"chroma_db/single/{collection_name}"

            # 步骤1: 清理旧存储
            if self.progress_callback:
                self.progress_callback({
                    'step': 1, 
                    'message': f"准备存储空间: {collection_name}"
                })
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)

            # 步骤2: 加载文档
            if self.progress_callback:
                self.progress_callback({
                    'step': 2,
                    'message': "解析文档结构..."
                })
            if path.endswith('.pdf'):
                loader = UnstructuredPDFLoader(file_path=path)
            elif path.endswith(('.doc', '.docx')):
                loader = UnstructuredWordDocumentLoader(file_path=path)
            else:
                raise ValueError(f"Unsupported file type: {path}")
            data = loader.load()
            full_text = "\n".join([doc.page_content for doc in data])
            self.raw_texts[path] = full_text

            # 步骤3: 文本分块
            if self.progress_callback:
                self.progress_callback({
                    'step': 3,
                    'message': "分割文本内容..."
                })
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(data)

            # 步骤4: 创建向量存储
            if self.progress_callback:
                self.progress_callback({
                    'step': 4,
                    'message': "生成向量嵌入..."
                })
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="deepseek-r1:8b"),
                collection_name=collection_name,
                persist_directory=persist_dir
            )

            self.single_docs[path] = vector_db

            # 完成通知
            if self.progress_callback:
                self.progress_callback({
                    'step': 5,
                    'message': "完成文档加载"
                })

        except Exception as e:
            error_msg = f"无法加载 {os.path.basename(path)}: {str(e)}"
            # 错误通知
            if self.progress_callback:
                self.progress_callback({
                    'type': 'error',
                    'message': error_msg
                })
            QMessageBox.critical(self.main_window, "文档加载错误", error_msg)
            raise
    
    def get_raw_text(self, path):
        return self.raw_texts.get(path, "")

    def _add_to_collection(self, paths):
        """多文档集合存储"""
        # 确保输入是列表且元素是字符串
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("Invalid paths format")
        
        # 删除旧集合（如果存在）
        if self.collection_db:
            try:
                self.collection_db.delete_collection()
                print("已清除旧集合")
            except Exception as e:
                print(f"清除旧集合失败: {str(e)}")

        # 创建新集合
        #collection_name = self._normalize_collection_name("multi_doc_rag")
        #self.collection_db = Chroma(
            #embedding_function=OllamaEmbeddings(model="deepseek-r1:8b"),
            #persist_directory="multi_doc_db/",
            #collection_name=collection_name
        #)
        # 创建持久化目录(确保存在)
        persist_dir = "multi_doc_db"
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path="multi_doc_db/")
        # 使用固定集合名称和持久化路径
        collection_name = self._normalize_collection_name("multi_doc_rag")
        
        # 先尝试获取已有集合
        try:
            # 尝试获取已有集合
            collection = client.get_collection(collection_name)
        except Exception as e:
            # 创建新集合
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        # 合并所有文档内容
        all_chunks = []
        existing_sources = set()
        if collection.count() > 0:
            metadatas = collection.get()["metadatas"]  # 获取元数据列表
            existing_sources = {m.get("source", "") for m in metadatas}  # 遍历每个元数据字典
        for path in paths:
            if path in existing_sources:
                continue
            # 添加路径有效性检查
            if not os.path.isfile(path):
                print(f"Invalid file path: {path}")
                continue

            if path in self.single_docs:
                print(f"Document already loaded: {path}")
                continue
            
            # 加载文档并分块
            if path.endswith('.pdf'):
                loader = UnstructuredPDFLoader(file_path=path)
            elif path.endswith(('.doc', '.docx')):
                loader = UnstructuredWordDocumentLoader(file_path=path)
            else:
                print(f"Unsupported file type: {path}")
                continue
            data = loader.load()

            # 文本分块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, 
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(data)

            # 添加元数据标识来源
            for chunk in chunks:
                # 确保元数据格式正确
                if not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata.update({
                    "source": os.path.basename(path),
                    "doc_id": str(hash(path)),
                    "timestamp": datetime.now().isoformat()
                })
                # 移除非法字符
            for key in list(chunk.metadata.keys()):
                if not isinstance(key, str):
                    del chunk.metadata[key]
            # 合并所有chunks
            all_chunks.extend(chunks)

            # 记录单个文档的集合引用
            self.single_docs[path] = self.collection_db

        # 批量添加所有chunks
        if all_chunks:
            collection.add(
                documents=[c.page_content for c in all_chunks],
                metadatas=[c.metadata for c in all_chunks],
                ids=[f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(all_chunks))]
            )
        self.current_collection = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model="deepseek-r1:8b")
        )
        print(f"成功添加 {len(all_chunks)} 个文档块到集合")
        self.current_collection = self.collection_db
        self.loaded_paths.extend(paths)

    def get_context(self, question, current_path=None):
        """获取上下文"""
        if self.current_mode == "collection":
            results = self.collection_db.similarity_search(question, k=3)
            context = "\n".join([r.page_content for r in results])
        else:
            if current_path not in self.single_docs:
                print(f"No document found for path: {current_path}")
                return ""
            results = self.single_docs[current_path].similarity_search(question, k=3)
            context = "\n".join([r.page_content for r in results])
        return context

    def _collection_retrieve(self, question):
        """多文档检索"""
        results = self.collection_db.collection("multi_docs").query(
            query_texts=[question],
            n_results=5
        )
        return "\n".join(results['documents'][0])

    def _single_retrieve(self, question, path):
        """单文档检索"""
        if path not in self.single_docs:
            return ""
        return self.single_docs[path].search(question)
    
    def set_current_document(self, path):
        """增强版文档切换方法"""
        try:
            if path not in self.single_docs:
                print(f"开始加载文档: {path}")
                self._add_single_doc(path)  # 确保加载完成
                
            # 获取集合引用
            target_collection = self.single_docs.get(path)
            
            if not target_collection:
                raise ValueError(f"文档集合初始化失败: {path}")
                
            # 显式连接集合
            if not hasattr(target_collection, '_collection'):
                print("检测到未连接集合，尝试重新连接...")
                target_collection._client = chromadb.PersistentClient(
                    path=target_collection._persist_directory
                )
                target_collection._collection = target_collection._client.get_collection(
                    target_collection._collection.name,
                    embedding_function=target_collection._embedding_function
                )
                
            self.current_collection = target_collection
            print(f"成功加载集合: {self.current_collection._collection.name}")
            
            # 强制验证
            if not self.verify_collection():
                raise RuntimeError("集合验证失败")
            
        except Exception as e:
            print(f"文档切换失败: {str(e)}")
            self.current_collection = None
            # 触发自动恢复
            self.rebuild_collection(path)

    def get_current_context(self, question):
        """获取当前活动文档的上下文"""
        if not self.current_collection:
            raise ValueError("未选择任何文档")
            
        return self.current_collection.similarity_search(
            question, 
            k=5 if self.current_mode == "collection" else 3
        )
    
    def get_questions_answers(self, path):
        return self.documents.get(path, [])

    def set_questions_answers(self, path, qa_list):
        self.documents[path] = qa_list

    def get_combined_context(self, question):
    # 添加空值保护
        if not question or not isinstance(question, str):
            return ""    
        # 检查当前集合是否为空，如果为空则返回空字符串
        if not self.current_collection or not self.main_window.active_links:
            return ""  
        
        try:
            if self.current_mode == "collection":
                if not self.collection_db:
                    return ""
                
                # 获取所有文档来源
                all_sources = [os.path.basename(source) for source in self.single_docs.keys()]
                # 获取关联文档的源文件名
                sources = [os.path.basename(path) for path in self.main_window.active_links]
                # 使用MMR算法提升结果多样性
                retriever = self.collection_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        'k': 6,
                        'fetch_k': 20,
                        'lambda_mult': 0.5,
                        'filter': {'source': {'$in': sources}}  # 关键过滤条件
                    }
                )
                docs = retriever.invoke(question)
                docs = retriever.invoke("测试问题")
                print([os.path.basename(doc.metadata["source"]) for doc in docs])
                sorted_docs = sorted(
                    docs,
                    key=lambda x: (
                        datetime.now() - datetime.fromisoformat(x.metadata["timestamp"])
                    ).total_seconds(),
                    reverse=True
                )[:4]  # 取时间最新的前4个结果

                # 按文档分组并标记来源
                context_dict = defaultdict(list)
                for doc in sorted_docs:
                    source = os.path.basename(doc.metadata.get("source", "未知文档"))
                    context_dict[source].append(doc.page_content[:500] + "...")  # 截断长文本

                # 构建带来源标识的上下文
                context_str = ""
                for source, contents in context_dict.items():
                    context_str += f"\n\n### 来自《{source}》的相关内容："
                    context_str += "\n".join([f"- {c}" for c in contents[:3]])  # 每篇取前3个相关段落
                
                return context_str
            else:
                # 单文档处理逻辑
                results = self.current_collection.similarity_search(
                    question,
                    k=5
                )
                
                # 过滤无效结果
                valid_results = []
                for result in results:
                    if hasattr(result, 'page_content') and hasattr(result, 'metadata'):
                        page_content = getattr(result, 'page_content')
                        metadata = getattr(result, 'metadata')
                        if isinstance(page_content, str) and isinstance(metadata, dict):
                            valid_results.append(f"[来源: {metadata.get('source','未知')}]\n{page_content}")
                        
                return "\n\n".join(valid_results[:3])  # 最多返回3个相关段落
    
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""
        
class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.parent_window = parent  # 保存父窗口引用
        self.is_3d = False  # 初始状态为2D
        
        # 创建2D/3D切换按钮
        self.toggle_action = QAction('🌐 3D/2D', self)
        self.toggle_action.setCheckable(True)
        self.toggle_action.setChecked(False)
        self.toggle_action.triggered.connect(self.toggle_3d)
        
        # 在适当位置插入按钮（这里放在保存按钮之后）
        self.insertAction(self.actions()[6], self.toggle_action)
    
    def toggle_3d(self, checked):
        """切换2D/3D视图"""
        self.is_3d = checked
        self.toggle_action.setText('🌐2D' if checked else '🌐3D')
        
        # 获取当前活动标签页
        current_index = self.parent_window.tabs.currentIndex()
        self.parent_window.current_3d_states[current_index] = checked
        
        if current_index == 0:  # 柱状图标签页
            self.parent_window.redraw_bar_chart(checked)
        elif current_index == 2:  # 词性分类标签页
            self.parent_window.redraw_pos_chart(checked)
        elif current_index == 3:  # 实体统计标签页
            self.parent_window.redraw_entity_chart(checked)
        elif current_index == 5:  # 如果有其他需要3D的图表
            pass  # 可以添加其他图表的3D重绘逻辑    
    
    def update_toolbar(self, index):
        # 恢复该标签页的3D状态
        self.toggle_action.setChecked(self.parent_window.current_3d_states.get(index, False))
        self.toggle_3d(self.parent_window.current_3d_states.get(index, False))

class WordFrequencyWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("词频分析")
        self.current_3d_states = {
            0: False,  # 柱状图
            2: False,  # 词性分类
            3: False   # 实体统计
        }
        self.setGeometry(200, 200, 1200, 800)
        self.parent = parent
        font_paths = [
            (r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\SimHei.ttf", 'SimHei'),
            (r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\ipaex\ipaexg.ttf", 'IPAexGothic')
        ]
        
        for path, font_name in font_paths:
            if os.path.exists(path):
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name]
                break
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.figure1 = Figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.figure4 = Figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.figure5 = Figure()
        self.canvas5 = FigureCanvas(self.figure5)
        
        self.canvas5.mpl_connect('pick_event', self.on_node_click)
        # 工具栏
        self.toolbar = CustomToolbar(self.canvas1, self)
        
        # 布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
        # 添加标签页
        self.tabs.addTab(self.canvas1, "柱状图")
        self.tabs.addTab(self.canvas2, "词云图")
        self.tabs.addTab(self.canvas3, "词性分类")
        self.tabs.addTab(self.canvas4, "实体统计")
        self.tabs.addTab(self.canvas5, "概念关系")

        
        # 连接标签切换事件
        self.tabs.currentChanged.connect(self.update_toolbar)
        self.toolbar.actions()[0].setVisible(False)

    def get_font(self):
        """获取当前语言对应的字体"""
        if self.parent.is_chinese:
            return FontProperties(fname=r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\SimHei.ttf", size=8)
        elif self.parent.is_japanese:
            return FontProperties(fname=r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\ipaex\ipaexg.ttf", size=7)
        return FontProperties(family='sans-serif', size=9)

    def on_node_click(self, event):
        """节点点击事件"""
        node = event.artist.get_label()
        QMessageBox.information(self, 
            "节点详情",
            f"概念: {node}\n关联数量: {self.graph.degree[node]}"
        )

    def update_toolbar(self, index):
        """工具栏方法"""
        # 移除旧的工具栏
        self.layout().removeWidget(self.toolbar)
        self.toolbar.deleteLater()
        
        # 创建新的自定义工具栏
        if index == 0:
            canvas = self.canvas1
        elif index == 1:
            canvas = self.canvas2
        elif index == 2:
            canvas = self.canvas3
        elif index == 3:
            canvas = self.canvas4
        else:
            canvas = self.canvas5
            
        self.toolbar = CustomToolbar(canvas, self)
        
        # 根据当前标签页决定是否显示3D按钮
        show_3d_button = index in [0,2,3]  # 只在柱状图标签页显示
        self.toolbar.toggle_action.setVisible(show_3d_button)
        
        # 添加新的工具栏到布局
        self.layout().insertWidget(0, self.toolbar)  # 插入到顶部

    def create_click_handler(self, words):
        def handler(event):
            current_3d = self.current_3d_states[self.tabs.currentIndex()]
            
            # 2D模式处理
            if not current_3d and event.inaxes == self.figure1.axes[0]:
                y_coord = event.ydata
                idx = int(y_coord + 0.5)
                if 0 <= idx < len(words):
                    self.show_paragraph_distribution(words[idx])
            
            # 3D模式处理
            elif current_3d and event.inaxes == self.figure1.axes[0]:
                x, y = event.xdata, event.ydata
                nearest_idx = np.argmin(np.abs(self.figure1.axes[0].get_xticks() - x))
                if 0 <= nearest_idx < len(words):
                    self.show_paragraph_distribution(words[nearest_idx])
                    
        return handler
    
    def show_paragraph_distribution(self, word):
        """显示词汇段落分布详情"""

        if isinstance(word, (np.generic)):
            word = str(word.item())
        elif not isinstance(word, str):
            word = str(word)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"“{word}”的段落分布")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # 显示统计摘要
        summary = QLabel(f"“{word}”在 {self.parent.word_para_counts.get(word, 0)} 个段落中出现")
        layout.addWidget(summary)
        
        # 段落列表
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        # 获取原始段落数据
        raw_text = self.parent.doc_manager.get_raw_text(self.parent.get_current_file_path())
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
        
        # 高亮显示包含词汇的段落
        highlight_css = """
            span.highlight { 
                background-color: yellow; 
                font-weight: bold;
            }
        """
        content = []
        for idx, para in enumerate(paragraphs[:50]):  # 只显示前50个段落
            if word in para:
                marked_para = para.replace(word, f'<span class="highlight">{word}</span>')
                content.append(f"<b>段落 {idx+1}:</b><br>{marked_para}<br><hr>")
        
        text_edit.setHtml(f"<style>{highlight_css}</style>" + "<br>".join(content))
        layout.addWidget(text_edit)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def plot_bar(self, word_counts, is_chinese):
        """绘制柱状图（兼容中日英文）"""
        self.figure1.clear()
        ax = self.figure1.add_subplot(111)
        
        # 数据分级处理
        words = [str(wc[0]) for wc in word_counts][:20]  # 只取前20个高频词汇
        counts = [wc[1] for wc in word_counts][:20]
        max_count = max(counts) if counts else 1
        
        # 现代渐变色方案
        cmap = LinearSegmentedColormap.from_list("custom", ['#1ABC9C', '#3498DB'])
        colors = [cmap(i/(len(words)-1)) for i in range(len(words))]
        
        # 绘制高级柱状图
        bars = ax.barh(words, counts, 
                      color=colors, 
                      height=0.68,
                      edgecolor='#34495E',
                      linewidth=0.8,
                      alpha=0.85)
        
        # 动态阴影效果
        for bar in bars:
            bar.set_path_effects([
                patheffects.withSimplePatchShadow(
                    offset=(2,-2), 
                    alpha=0.3,
                    shadow_rgbFace='#FFFFFF'
                )
            ])
        
        # 智能数据标注
        for i, (word, count) in enumerate(zip(words, counts)):
            ax.text(
                count + max_count*0.02, i, 
                f'{count:,}', 
                va='center',
                fontsize=10,
                color='#2C3E50',
                fontweight='medium'
            )
        
        # 自适应布局参数
        ax.set_xlim(0, max_count * 1.15)
        ax.invert_yaxis()
        
        # 专业样式配置
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.grid(axis='x', linestyle=':', alpha=0.6, color='#EAECEE')
        
        # 动态多语言标题
        title_settings = {
            'chinese': ('高频词汇统计（TOP 20）', '出现次数'),
            'japanese': ('単語頻度トップ20', '出現回数'),
            'english': ('Top 20 Frequent Words', 'Frequency')
        }
        current_font = plt.rcParams['font.sans-serif'][0]
        if 'SimHei' in current_font:
            lang = 'chinese'
        elif 'IPAex' in current_font:
            lang = 'japanese'  # 新增日语判断
        else:
            lang = 'english'
        
        ax.set_xlabel(title_settings[lang][1], 
                    fontsize=12, 
                    labelpad=10,
                    color='#34495E')
        ax.set_title(title_settings[lang][0], 
                    fontsize=14,
                    pad=20,
                    color='#2C3E50',
                    fontweight='semibold')
        
        # 响应式字体调整
        plt.rcParams.update({
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
        self.figure1.tight_layout(pad=3.0)

        # 添加点击事件
        def on_click(event):
            if event.inaxes == ax:
                # 通过坐标转换获取索引
                y_coord = event.ydata
                idx = int(y_coord + 0.5)  # 取最近的整数索引
                if 0 <= idx < len(words):
                    word = words[idx]
                    self.show_paragraph_distribution(word)

        if self.canvas1:
            self.canvas1.mpl_connect('button_press_event', on_click)
            self.canvas1.draw()

    def redraw_bar_chart(self, is_3d):
        """根据3D状态重绘柱状图"""
        self.figure1.clear()
        
        if is_3d:
            ax = self.figure1.add_subplot(111, projection='3d')
            word_counts = self.parent.word_counts

            # 提取三维数据
            words = [str(wc[0]) for wc in word_counts][:20]
            counts = [wc[1] for wc in word_counts][:20]
            spans = [wc[2] for wc in word_counts][:20]
            xpos = np.arange(len(words))  # 柱子x坐标数组
            
            # 创建颜色映射（文档分布广度）
            cmap = plt.get_cmap('viridis')
            edge_color = '#f5f5f5'  # 浅灰边框
            alpha = 0.85
            max_span = max(spans) if spans else 1
            colors = [cmap(span/max_span) for span in spans]

            # 三维柱体参数
            xpos = np.arange(len(words))
            ypos = np.zeros(len(words))
            dx = dy = 0.8
            
            # 绘制三维柱状图
            bars = ax.bar3d(
                xpos, ypos, np.zeros(len(words)),
                dx, dy, counts,
                color=colors,
                alpha=alpha,          # 增加透明度
                edgecolor=edge_color, # 柔化边框
                linewidth=0.5
            )

            # 设置三维坐标轴
            ax.set_xticks(xpos)
            ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([])
            ax.set_zlabel('Frequency', labelpad=12)
            
            # 添加颜色条
            norm = plt.Normalize(0, max_span)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = self.figure1.colorbar(sm, ax=ax, pad=0.1)
            cbar.outline.set_edgecolor('#cccccc')  # 颜色条边框柔化
            cbar.set_label('分布广度（段落数）', rotation=270, labelpad=15)

            # 优化视角
            ax.view_init(elev=28, azim=-45)
            ax.set_title('3D 高频词汇统计', pad=15)

            # 双击处理相关变量
            self.last_click_time = 0  # 记录上次点击时间
            self.double_click_threshold = 300  # 双击间隔阈值（毫秒）

            def on_click_3d(event):
                # 仅处理左键双击
                if event.button != 1:
                    return
                
                current_time = time.time() * 1000  # 转换为毫秒
                time_diff = current_time - self.last_click_time
                
                # 判断是否双击
                if time_diff < self.double_click_threshold:
                    # 执行双击操作
                    if event.inaxes == ax:
                        x2d, y2d = event.x, event.y
                        x3d = ax.get_xticks()
                        # 创建坐标映射矩阵
                        inv_proj = ax.get_proj()
                        view_matrix = ax.viewLim
                        # 计算每个柱子的屏幕位置
                        screen_positions = [
                            ax.transData.transform((xi, 0)) 
                            for xi in x3d
                        ]
                        # 计算与点击位置的欧氏距离
                        distances = [
                            np.linalg.norm([x2d - pos[0], y2d - pos[1]])
                            for pos in screen_positions
                        ]
                        
                        nearest_idx = np.argmin(distances)
                        if 0 <= nearest_idx < len(words):
                            # 视觉反馈开始
                            ax.patch.set_facecolor('#F0F0F0')
                            self.canvas1.draw_idle()

                            word = words[nearest_idx]
                            self.show_paragraph_distribution(word)

                            # 延迟恢复
                            QTimer.singleShot(100, lambda: 
                                ax.patch.set_facecolor('white') or 
                                self.canvas1.draw_idle()
                            )
                            # 视觉反馈结束
                    
                    # 重置时间避免连续触发
                    self.last_click_time = 0
                else:
                    # 记录首次点击时间，保留默认旋转功能
                    self.last_click_time = current_time

            self.canvas1.mpl_connect('button_press_event', on_click_3d)
            
        else:
            # 原有2D绘图逻辑保持不变
            self.plot_bar(self.parent.word_counts, self.parent.is_chinese_flag)

        self.canvas1.draw()

    def plot_wordcloud(self, text, is_chinese=True, mask_image_path=None):
        """生成词云图（兼容中日英文）"""
        self.figure2.clear()
        ax = self.figure2.add_subplot(111)
        
        # 根据当前字体判断语言类型
        current_font = plt.rcParams['font.sans-serif'][0]
        if 'SimHei' in current_font:
            lang = 'zh'
            title = '词云图'
            font_path = r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\SimHei.ttf"
            regexp = None
        elif 'IPAex' in current_font:
            lang = 'ja'
            title = 'ワードクラウド'
            font_path = r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\ipaex\ipaexg.ttf"
            regexp = r"[\w']+"
        else:
            lang = 'en'
            title = 'Word Cloud'
            font_path = None
            regexp = r"\w[\w']+"

        # 验证字体文件存在
        if font_path and not os.path.exists(font_path):
            print(f"警告：字体文件不存在 - {font_path}")
            font_path = None

        # 词云参数
        wordcloud_params = {
            'background_color': '#F8F9FA',  # 浅白灰背景
            'width': 1200,
            'height': 800,
            'max_words': 200,
            'collocations': False,  # 解决英文重复问题
            'regexp': r"\w[\w']+" if not is_chinese else None,  # 英文单词匹配
            'contour_width': 3,
            'contour_color': 'steelblue',
            'font_path': font_path
        }
        
        # 加载掩膜图像
        mask = None
        if mask_image_path and os.path.exists(mask_image_path):
            mask = np.array(Image.open(mask_image_path))
            wordcloud_params['mask'] = mask
            wordcloud_params['color_func'] = ImageColorGenerator(mask)
        
        
        if font_path:
            wordcloud_params['font_path'] = font_path
        
        if mask is not None:
            wordcloud_params['mask'] = mask
            image_colors = ImageColorGenerator(mask)
            wordcloud_params['color_func'] = image_colors
        
        wordcloud = WordCloud(**wordcloud_params).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('词云图' if is_chinese else 'Word Cloud', fontsize=24, color='#333333')  # 深灰色标题
        
        if self.canvas2:
            self.canvas2.draw()

    def plot_pos(self, pos_stats, is_chinese):
        self.figure3.clear()
        ax = self.figure3.add_subplot(111)
        
        current_font = plt.rcParams['font.sans-serif'][0]
        is_japanese = 'IPAex' in current_font
        
        # 多语言配置
        config = {
            'zh': {
                'labels': {'Content': '内容词', 'Function': '功能词', 'Other': '其他'},
                'title': "词性分类统计",
                'no_data': "无词性数据"
            },
            'ja': {
                'labels': {'Content': '内容語', 'Function': '機能語', 'Other': 'その他'},
                'title': "品詞分類統計",
                'no_data': "データなし"
            },
            'en': {
                'labels': {'Content': 'Content', 'Function': 'Function', 'Other': 'Other'},
                'title': "POS Statistics",
                'no_data': "No POS Data"
            }
        }
        # 获取当前配置
        lang = 'ja' if is_japanese else 'zh' if is_chinese else 'en'
        conf = config[lang]
        
        # 标签处理
        labels = [conf['labels'].get(k, k) for k in pos_stats.keys()]
        values = list(pos_stats.values())
        
        print(f"【绘图诊断】POS统计: {pos_stats}")
        print(f"【绘图诊断】Labels: {labels}")
        print(f"【绘图诊断】Values: {values}")
        
        if sum(values) == 0:  # 空数据检查
            ax.text(0.5, 0.5, "无词性数据" if is_chinese else "No POS Data",
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # 生成颜色
            colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#DCBCBC', '#9370DB']
            explode = [0.03] * len(values)
            
            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors[:len(labels)],
                explode=explode,
                wedgeprops={'edgecolor': 'white'},
                textprops={'fontsize': 10}
            )
            
            # 调整标签样式
            for text in texts:
                text.set_fontsize(12)
                text.set_color('#2C3E50')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 添加中心圆
            center_circle = plt.Circle((0, 0), 0.7, color='white')
            ax.add_artist(center_circle)
            
            # 设置标题
            ax.set_title(conf['title'], pad=20, fontsize=14)
            ax.axis('equal')
            
            # 添加图例
            ax.legend(wedges, labels,
                     loc="center left",
                     bbox_to_anchor=(1, 0, 0.5, 1),
                     frameon=False)

        self.figure3.tight_layout()
        if self.canvas3:
            self.canvas3.draw()

    def redraw_pos_chart(self, is_3d):
        """重绘词性分类图表"""
        self.figure3.clear()
        
        if is_3d:
            # 3D柱状图实现
            ax = self.figure3.add_subplot(111, projection='3d')
            pos_stats = self.parent.pos_stats
            categories = list(pos_stats.keys())
            values = list(pos_stats.values())
                        
            # 创建3D柱状图
            xpos = range(len(categories))
           
            bars = ax.bar3d(xpos, [0]*len(categories), [0]*len(categories),
                            0.8, 0.8, values,
                            color='#55A868',
                            alpha=0.8,
                            edgecolor='w')
            
            # 标签配置
            ax.set_xticks(xpos)
            ax.set_xticklabels(categories,  # 直接使用原始标签
                         rotation=45 if self.parent.is_chinese else 60,
                         ha='right',
                         va='top',
                         fontsize=9 if self.parent.is_chinese else 8)
        
            # 调整布局
            self.figure3.subplots_adjust(
                left=0.3 if self.parent.is_chinese else 0.35,
                right=0.95,
                bottom=0.2
            )
            # 视角优化
            ax.view_init(elev=28, azim=-45)
            ax.set_zlabel('出现次数' if self.parent.is_chinese else 'Count', labelpad=15)
            ax.set_title('3D 词性分布', fontsize=12)
        
            
        else:
            # 原有2D饼图逻辑
            self.plot_pos(self.parent.pos_stats, self.parent.is_chinese)
            
        self.canvas3.draw()

    def plot_entities(self, entities, is_chinese):
        print(f"【实体绘图诊断】接收的实体数据: {entities}")
        self.figure4.clear()
        ax = self.figure4.add_subplot(111)
        
        # 动态判断语言类型
        current_font = plt.rcParams['font.sans-serif'][0]
        is_japanese = 'IPAex' in current_font

        if not entities:
            no_data_msg = "データなし" if is_japanese else \
                     "无实体数据" if is_chinese else "No entity data"
            ax.text(0.5, 0.5, no_data_msg, 
                    ha='center', va='center', fontsize=14,
                    fontproperties=FontProperties(fname=self.ja_font_path) if is_japanese else None)
            ax.axis('off')
            self.canvas4.draw()
            return
        
        # 统一排序逻辑
        sorted_entities = sorted(entities.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:15]
        
        # 双语标签处理
        if is_japanese:
            labels = [ENTITY_MAPPING_JA.get(k, k) for k, v in sorted_entities]
        else:
            labels = [ENTITY_MAPPING.get(k, k) if is_chinese else k 
                    for k, v in sorted_entities]
        
        counts = [v for k, v in sorted_entities]
        
        # 颜色方案
        colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
        
        # 绘制高级条形图
        bars = ax.barh(labels, counts, 
                      color=colors, 
                      edgecolor='#34495E',
                      height=0.7,
                      linewidth=0.8,
                      alpha=0.85)
        
        # 动态阴影效果
        for bar in bars:
            bar.set_path_effects([
                patheffects.withSimplePatchShadow(
                    offset=(2,-2), 
                    alpha=0.2,
                    rho=0.8
                )
            ])
        
        # 添加数据标签
        max_count = max(counts) if counts else 1
        for i, v in enumerate(counts):
            ax.text(v + max_count * 0.02, i, 
                    f"{v:,}", 
                    va='center',
                    fontsize=10,
                    color='#2C3E50',
                    fontweight='medium')
        
        # 自适应布局参数
        ax.set_xlim(0, max_count * 1.15)
        ax.invert_yaxis()
        
        # 专业样式配置
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        # 多语言标题配置
        title_map = {
            'ja': ('エンティティ統計', '出現回数'),
            'zh': ('实体类型统计', '出现次数'),
            'en': ('Entity Statistics', 'Count')
        }
        lang = 'ja' if is_japanese else 'zh' if is_chinese else 'en'
    
        
        # 设置标题和标签
        ax.set_xlabel(title_map[lang][1], 
                    labelpad=10, 
                    fontsize=12, 
                    color='#34495E',
                    fontproperties=FontProperties(fname=self.ja_font_path) if is_japanese else None)
        ax.set_title(title_map[lang][0], 
                    pad=20, 
                    fontsize=14, 
                    color='#2C3E50',
                    fontweight='semibold',
                    fontproperties=FontProperties(fname=self.ja_font_path) if is_japanese else None)
        
        # 响应式字体调整
        plt.rcParams.update({
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
        self.figure4.tight_layout(pad=3.0)
        if self.canvas4:
            self.canvas4.draw()

    def redraw_entity_chart(self, is_3d):
        """重绘实体统计图表"""
        self.figure4.clear()
        
        if is_3d:
            ax = self.figure4.add_subplot(111, projection='3d')
            entities = self.parent.entities
            sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:15]
            labels = [k for k, v in sorted_entities]
            values = [v for k, v in sorted_entities]

            # 创建3D柱状图
            xpos = range(len(labels))
            bars = ax.bar3d(xpos, [0]*len(labels), [0]*len(labels),
                            0.8, 0.8, values,
                            color=plt.cm.tab20c(np.linspace(0, 1, len(labels))),
                            alpha=0.8
            )

            # 标签处理
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels,  # 直接使用原始标签
                            rotation=50 if self.parent.is_chinese else 65,
                            ha='right',
                            va='top',
                            fontsize=9 if self.parent.is_chinese else 8,
                            fontproperties=self.get_font())
            
            # 动态调整布局
            max_label_len = max(len(str(l)) for l in labels)
            adjust_left = 0.25 + (max_label_len * 0.015)  # 根据标签长度动态调整
            self.figure4.subplots_adjust(left=adjust_left, right=0.95, bottom=0.15)
            
            # 视角优化
            ax.view_init(elev=25, azim=-50)
            ax.set_zlabel('出现次数' if self.parent.is_chinese else 'Count', labelpad=15)
            ax.set_title('3D 实体分布', fontsize=12)
            
        else:
            self.plot_entities(self.parent.entities, self.parent.is_chinese)
    
        self.canvas4.draw()

    def plot_relations(self, relations, is_chinese):
        """概念关系图绘制"""
        self.figure5.clear()
        ax = self.figure5.add_subplot(111)
        
        if not relations:
            ax.text(0.5, 0.5, "无关系数据" if is_chinese else "No relations", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.canvas5.draw()
            return

        # 创建有向图更符合语义关系
        G_directed = nx.DiGraph() if any(rel[1] for rel in relations) else nx.Graph()
        
        # 带权边处理
        edge_weights = defaultdict(int)
        edge_colors = {}
        node_types = {}  # 存储节点类型
        for rel in relations:
            if len(rel) >= 3:
                key = (rel[0], rel[2], rel[1])  # (subj, obj, relation)
                edge_weights[key] += 1
                edge_colors[key] = rel[1]
                
                # 假设每个关系的第一个元素是主体，第三个元素是客体，第二个元素是关系类型
                subj_type = 'entity'  # 这里可以扩展为从其他地方获取实体类型
                obj_type = 'entity'
                node_types[rel[0]] = subj_type
                node_types[rel[2]] = obj_type

        # 按权重筛选和排序
        top_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:20]
        for (subj, obj, rel), weight in top_edges:
            G_directed.add_edge(subj, obj, label=rel, weight=weight)

        # 将有向图转换为无向图以进行社区检测
        G_undirected = G_directed.to_undirected()

        # 社区检测
        partition = community_louvain.best_partition(G_undirected)
        unique_communities = set(partition.values())
        num_communities = len(unique_communities)
        community_color_map = {com: plt.cm.tab20(i % 20) for i, com in enumerate(unique_communities)}

        # 力导向布局
        pos = nx.spring_layout(G_directed, seed=42, k=0.5 / np.sqrt(len(G_directed)), iterations=50)

        # 节点尺寸动态调整
        node_size = np.clip(2500 / np.sqrt(len(G_directed)), 300, 1500)
        font_size = np.clip(14 - len(G_directed) // 20, 8, 12)
        
        # 边宽度分级处理
        max_weight = max([d['weight'] for u,v,d in G_directed.edges(data=True)])
        edge_width = [0.5 + 2*(d['weight']/max_weight) for u,v,d in G_directed.edges(data=True)]

        # 颜色映射增强
        unique_rels = list(set(edge_colors.values()))
        color_map = {rel: plt.cm.tab20(i%20) for i, rel in enumerate(unique_rels)}
        colors = [color_map[edge_colors[(u, v, G_directed[u][v]['label'])]] for u, v in G_directed.edges()]

        # 节点颜色区分
        type_to_color = {'entity': '#87CEEB', 'attribute': '#FFA07A'}
        base_colors = [to_rgb(type_to_color.get(node_types[node], '#C0C0C0')) for node in G_directed.nodes()]
        community_colors = [community_color_map[partition[node]] for node in G_directed.nodes()]

        # 合并社区颜色和类型颜色
        final_node_colors = []
        for base_color, community_color in zip(base_colors, community_colors):
            final_color = tuple(b * 0.6 + c * 0.4 for b, c in zip(base_color, community_color))
            final_node_colors.append(final_color)

        # 绘制优化
        nx.draw_networkx_nodes(
            G_directed, pos, ax=ax,
            node_size=node_size,
            node_color=final_node_colors,
            alpha=0.9,
            edgecolors='#404040',
            linewidths=0.8
        )
        
        nx.draw_networkx_edges(
            G_directed, pos, ax=ax,
            width=edge_width,
            edge_color=colors,
            alpha=0.7,
            arrowsize=18,
            connectionstyle='arc3,rad=0.1'  # 增加边曲率
        )

        # 标签动态偏移
        label_pos = {}
        for node, coords in pos.items():
            x_offset = 0.02 * (1 if hash(node) % 2 else -1)  # 随机偏移方向
            y_offset = 0.02 * (hash(node) % 3)
            label_pos[node] = (coords[0] + x_offset, coords[1] + y_offset)

        # 节点标签绘制
        current_font = plt.rcParams['font.sans-serif'][0]
        font_family = 'Arial'
        font_path = None

        if 'IPAex' in current_font:  # 日语字体
            font_family = 'IPAexGothic'
            font_path = r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\ipaex\ipaexg.ttf"
            if font_path:
                font_prop = FontProperties(fname=font_path)
                plt.register_font(font_prop)
                font_family = font_prop.get_name()
        elif 'SimHei' in current_font:  # 中文
            font_family = 'SimHei'

        text_items = nx.draw_networkx_labels(
            G_directed, label_pos, ax=ax,
            font_size=font_size,
            font_family=font_family,
            font_weight='bold',
            verticalalignment='baseline',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.65,
                boxstyle='round,pad=0.2'
            )
        )

        # 智能标签防重叠（增强版）
        if 'adjustText' in globals():
            adjust_params = {
                'only_move': {'points':'y', 'texts':'xy'},
                'autoalign': 'y',
                'force_text': (0.01, 0.02),
                'expand_text': (1.05, 1.2),
                'expand_points': (1.05, 1.2),
                'lim': 50
            }
            adjustText.adjust_text(list(text_items.values()), **adjust_params)

        # 图例增强
        legend_elements = [
            Line2D([0], [0], color=color_map[rel], lw=2, label=f"{rel} ({len([e for e in edge_colors.values() if e == rel])})")
            for rel in unique_rels[:15]  # 显示前15种关系
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=font_size-1,
            title='关系类型',
            title_fontsize=font_size,
            framealpha=0.7
        )

        # 添加社区图例
        community_legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f"Community {com}", markersize=10, markerfacecolor=community_color_map[com])
            for com in unique_communities
        ]
        ax.legend(
            handles=community_legend_elements,
            loc='lower left',
            fontsize=font_size-1,
            title='Communities',
            title_fontsize=font_size,
            framealpha=0.7
        )

        # 画布优化
        ax.set_facecolor('#FAFAFA')
        ax.collections[0].set_zorder(100)  # 节点置顶
        self.figure5.tight_layout(pad=3.0)
        self.canvas5.draw()


class ChatThread(QThread):
    new_text = Signal(str)  # 定义信号
    new_char = Signal(str)
    finished_with_response = Signal(str)  # 信号，传递最终的回答内容

    def __init__(self, prompt, messages, document_manager, model, paths = None, parent=None,context_session=None):
        super().__init__(parent)
        self.setPriority(QThread.HighPriority)
        self.prompt = prompt
        self.messages = messages
        self.document_manager = document_manager
        self.model = model
        self.paths = paths if paths is not None else []
        self._is_running = True
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()
        self._pause_flag = False
        self.text_buffer = ""
        self.final_response = ""
        self.buffer_threshold = 1  # 按字符数缓冲
        self.sentence_enders = set()  # 禁用句子结束符缓冲
        self.sentence_enders = {'.', '!', '?', '。', '！', '？'}  # 句子结束符
        self.context_session = context_session

    def run(self):
        final_response = ""  # 确保在所有代码路径前初始化
        self._is_running = True
        self.text_buffer = ""  #缓冲变量
        self.dynamic_threshold = 8  # 动态阈值初始化
        self.last_emit_time = QDateTime.currentDateTime()  # 时间记录初始化
        # 打印当前运行的路径
        try:
            base_messages = [
                {
                    'role': 'system', 
                    'content': f"当前对话上下文：{self.document_manager.get_conversation_context(self.context_session)}"
                },
                *self.messages
            ]
            self.final_response = final_response
            for path in self.paths:
                if not self._is_running:
                        break
                print('run', f'Processing: {path}')
                # 动态调整逻辑开始
                current_time = QDateTime.currentDateTime()
                time_diff = self.last_emit_time.msecsTo(current_time)
                
                # 自动调整阈值（50ms为临界值）
                if time_diff < 50:  # 发射过快
                    self.dynamic_threshold = min(20, self.dynamic_threshold + 2)
                else:  # 发射间隔正常
                    self.dynamic_threshold = max(8, self.dynamic_threshold - 1)
                
                self.last_emit_time = current_time
            
                # 假设 ollama.chat 支持流式输出，返回一个迭代器
                if path != '':  # 如果有文件路径，则加载文件
                    messages_with_context = base_messages.copy()

                    if is_image(path):# 文件是图片# 读取图片并转换为模型可以接受的格式# 这里的转换方法取决于模型的具体要求# 例如，将图片转换为字节流
                        messages_with_context.append({
                            'role': 'user',
                            'content': self.prompt,
                            'images': [path]
                        })
                        for response_chunk in ollama.generate(model=self.model, prompt=self.messages[-1]['content'],images = [self.path], stream=True):
                            text = response_chunk['response']
                            self.new_text.emit(text)
                    elif path.endswith('.pdf'):
                        # 打印处理 PDF 文件的提示
                        print(f"Processing file: {path}")
                        
                        # 使用 UnstructuredPDFLoader 加载 PDF 文件
                        loader = UnstructuredPDFLoader(file_path=path)
                        data = loader.load()

                        # 对文档进行分割和处理
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                        chunks = text_splitter.split_documents(data)
                        # file_name = os.path.basename(self.path)
                        # file_name = os.path.splitext(os.path.basename(self.path))[0]
                        #persist_directory = self.path+"chroma_db"

                        # 动态生成持久化目录名，避免冲突
                        persist_directory = f"{os.path.splitext(path)[0]}_chroma_db_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            
                        # 定义一个函数来处理无法删除的文件
                        def remove_readonly(func, path, _):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                            
                        # 如果目录存在，强制删除它
                        if os.path.exists(persist_directory):
                            try:
                                shutil.rmtree(persist_directory, onexc=remove_readonly)
                            except PermissionError as e:
                                print(f"PermissionError: {e}. Retrying after 5 seconds...")
                                time.sleep(5)
                                shutil.rmtree(persist_directory, onexc=remove_readonly)

                        # 创建向量数据库 需要改进 2024年12月18日
                        vector_db = Chroma.from_documents(
                            documents=chunks,
                            embedding=OllamaEmbeddings(model="deepseek-r1:8b"),
                            collection_name="local-rag",
                            persist_directory=persist_directory
                        )

                        # 定义查询提示模板
                        QUERY_PROMPT = PromptTemplate(
                            input_variables=["question"],
                            template="""You are an AI language model assistant. Your task is to generate five
                        different versions of the given user question to retrieve relevant documents from
                        a vector database. By generating multiple perspectives on the user question, your
                        goal is to help the user overcome some of the limitations of the distance-based
                        similarity search. Provide these alternative questions separated by newlines.
                        Consider different ways to phrase the question, ask for synonyms, 
                        or provide more specific details that could lead to better retrieval results.
                        Original question: {question}
                        Reformulated questions:""",
                        )

                        llm = ChatOllama(model=self.model)

                        # 在创建检索器时添加校验逻辑
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                        chunks = text_splitter.split_documents(data)

                        # 动态计算最大可用结果数
                        max_results = min(4, len(chunks))  # 取4和实际块数的较小值                   
                        # 创建检索器
                        retriever = MultiQueryRetriever.from_llm(
                            vector_db.as_retriever(search_kwargs={"k": max_results}),
                            llm,
                            prompt=QUERY_PROMPT,

                        )

                        # 定义RAG提示模板
                        template = """基于以下结构化上下文回答问题时，请遵循科学论文解析规范：
                        <上下文>
                        {context}
                        </上下文>

                        <应答规范>
                        1. 答案结构：结论先行→证据支撑→方法论说明
                        2. 引证格式：[页码]标注原文位置 
                        3. 不确定性处理：对矛盾信息进行概率化表述
                        4. 可视化建议：对复杂数据给出图表绘制方案
                        5. 后续追问：生成3个深度研究问题
                        6. 默认情况用中文回答，如果问题中提出用别的语言，请选择最合适的语言进行回答。
                        

                        当前问题：{question}"""
                        prompt = ChatPromptTemplate.from_template(template)

                            # 构建链式调用
                        chain = (
                            {"context": retriever, "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        QMetaObject.invokeMethod(self.parent(), 
                                                "_start_stream_animation",
                                                Qt.QueuedConnection)
                        
                        # 将文档内容注入上下文
                        doc_content = "\n".join([d.page_content for d in data])
                        messages_with_context.append({
                            'role': 'system',
                            'content': f"当前文档内容：{doc_content[:2000]}..."  # 截断处理
                        })
                        for token in chain.stream({"question": self.messages[-1]['content']}):
                            self._check_pause()
                            print(token, end='', flush=True)
                            for char in token:  # 逐字符处理
                                self.text_buffer += char
                                if len(self.text_buffer) >= self.buffer_threshold:
                                    self.new_text.emit(self.text_buffer)
                                    self.text_buffer = ""
                                    # 添加微小延迟保证流畅性
                                    time.sleep(0.02)  # 调整这个值控制输出速度
                            # 确保清空缓冲区
                            if self.text_buffer:
                                self.new_text.emit(self.text_buffer)
                                self.text_buffer = ""

                    elif path.endswith(('.doc', '.docx')):
                            
                        loader = UnstructuredWordDocumentLoader(file_path=path)
                        data = loader.load()

                            # 对文档进行分割和处理
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                        chunks = text_splitter.split_documents(data)

                        persist_directory = f"{os.path.splitext(path)[0]}_chroma_db_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
                            # 定义一个函数来处理无法删除的文件
                        def remove_readonly(func, path, _):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                                    
                            # 如果目录存在，强制删除它
                        if os.path.exists(persist_directory):
                            try:
                                shutil.rmtree(persist_directory, onexc=remove_readonly)
                            except PermissionError as e:
                                print(f"PermissionError: {e}. Retrying after 5 seconds...")
                                time.sleep(5)
                                shutil.rmtree(persist_directory, onexc=remove_readonly)
                        
                            # 创建向量数据库
                        vector_db = Chroma.from_documents(
                            documents=chunks,
                            embedding=OllamaEmbeddings(model="deepseek-r1:8b"),
                            collection_name="local-rag",
                            persist_directory=persist_directory
                        )

                            # 定义查询提示模板
                        QUERY_PROMPT = PromptTemplate(
                            input_variables=["question"],
                            template="""You are an AI language model assistant. Your task is to generate five
                        different versions of the given user question to retrieve relevant documents from
                        a vector database. By generating multiple perspectives on the user question, your
                        goal is to help the user overcome some of the limitations of the distance-based
                        similarity search. Provide these alternative questions separated by newlines.
                        Original question: {question}""",
                        )

                        llm = ChatOllama(model=self.model)
                                            
                            # 创建检索器
                        retriever = MultiQueryRetriever.from_llm(
                            vector_db.as_retriever(),
                            llm,
                            prompt=QUERY_PROMPT
                        )

                        # 定义RAG提示模板
                        template = """Answer the question based ONLY on the following context:
                        {context}
                        Question: {question}
                        """
                        prompt = ChatPromptTemplate.from_template(template)

                        # 构建链式调用
                        chain = (
                            {"context": retriever, "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )

                        for token in chain.stream({"question": self.messages[-1]['content']}):
                            self._check_pause()
                            print(token, end='', flush=True)
                            for char in token:  # 逐字符处理
                                self.text_buffer += char
                                if len(self.text_buffer) >= self.buffer_threshold:
                                    self.new_text.emit(self.text_buffer)
                                    self.text_buffer = ""
                                    # 添加微小延迟保证流畅性
                                    time.sleep(0.02)  # 调整这个值控制输出速度
                            # 确保清空缓冲区
                            if self.text_buffer:
                                self.new_text.emit(self.text_buffer)
                                self.text_buffer = ""
                    else:    
                        combined_context = self.document_manager.get_combined_context(self.prompt)

                        if combined_context:
                            # 定义RAG提示模板
                            template = """Answer the question based ONLY on the following context:
                            {context}
                            Question: {question}
                            """
                            prompt = ChatPromptTemplate.from_template(template)

                            llm = ChatOllama(model=self.model)
                            
                            # 构建链式调用
                            chain = (
                                {"context": combined_context, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                        
                        else:
                            for response_chunk in ollama.chat(model=self.model, messages=self.messages,stream=True):
                                self._check_pause()  # 暂停检查点
                                text = response_chunk['message']['content']
                                self.new_text.emit(text)
        finally:
            if self.text_buffer:
                self.new_text.emit(self.text_buffer)
            QMetaObject.invokeMethod(self.parent(),
                                    "_stop_stream_animation",
                                    Qt.QueuedConnection)
            self._is_running = False
            self.finished_with_response.emit(final_response)  # 发射带有最终回答内容的信号
            self.finished.emit()

    def pause(self):
        with QMutexLocker(self._mutex):
            print("暂停请求已接收")
            self._pause_flag = True

    def stop(self):
        with QMutexLocker(self._mutex):
            self._is_running = False
            self.resume()  # 确保唤醒等待的线程
        self.quit()  # 请求线程退出事件循环
        self.wait(2000)  # 等待最多2秒让线程退出

    def resume(self):
        with QMutexLocker(self._mutex):
            print("恢复请求已接收")
            self._pause_flag = False
            self._pause_condition.wakeAll()

    def _check_pause(self):
        with QMutexLocker(self._mutex):
            while self._pause_flag:
                self._pause_condition.wait(self._mutex)

class ChatLocalAndPersistent(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.messages = {}   
        self.model = ""
        self.input_text = ""  # 初始化属性
        self.role = ""
        self.resize(1024, 600)  # 设置窗口尺寸为1024*600  
        self.qm_files = []
        self.path = ''  # 加载文件路径
        self.is_paused = False
        self.current_chat_thread = None  # 当前线程引用
        # 筛选出.qm文件        
        # self.output_text_list=[]
        self.show_text = ''
        self.text_labels = {
            'model': 'Model',
            'role': 'Role',
            'timestamp': 'Timestamp',
            'input_text': 'Input Text',
            'output_text': 'Output Text'
        }
        self.setStyleSheet("""
            QLabel#doc_info {
            font-family: 'Microsoft YaHei';
            font-size: 12px;
            color: #666;
            }
        """)
        self.active_links = []  # 初始化
        self.link_indicator = QLabel("已关联0篇文档") 
        self.current_file = None  # 当前文件追踪
        self.init_ui()
        self.setLanguage()
        self.doc_manager = DocumentManager(self)  # 文档管理器 
        self.style_initialized = False
        self.pending_updates = []
        self.update_timer = QTimer()
        self.word_counts = []  # 初始化词频数据存储
        self.is_chinese_flag = False  # 添加语言状态标志
        self.ltp_model_path = r"C:/Users/86157/Desktop/env/LTP/base1"  # 修改为实际的模型路径
        # 初始化LTP对象
        try:
            self.ltp = LTP(path=self.ltp_model_path)
        except FileNotFoundError as e:
            print(f"无法找到LTP模型文件: {e}")
            self.ltp = None
        self.update_timer.timeout.connect(self._flush_updates)
        # 实体映射
        self.entity_mapping = ENTITY_MAPPING

        self.zh_stop_words = {'的', '是', '在', '了', '和', '有', '这', '为', '也', 
                         '就', '要', '与', '等', '对', '中', '或', '日', '月', '年',
                         '第','级别','等','而', '但', '则', '且', '又', '再', '已', 
                         '将', '还', '因', '其'}
    
        self.en_stop_words = {
            'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'it', 'with',
            'for', 'on', 'this', 'be', 'are', 'as', 'at', 'by', 'from'
            'then', 'than', 'such', 'some', 'so', 'nor', 'not', 'into', 'onto', 'off'}
        # 中文词性到类别的映射
        self.zh_pos_mapping = {
            '实词': ['n', 'v', 'a', 'vn', 'vd', 'ad', 'an', 
                   't', 'b', 'i', 'j', 'l', 'z','geo', 'GEO_TERM'],  # 地质领域常见标签
            '虚词': ['c', 'u', 'p', 'm', 'q', 'r', 'd', 'xc', 'f']
        }

        # 添加特殊领域标签说明
        self.zh_pos_explain = {
            't': '时间词',       # 如"年代"
            'b': '区别词',       # 如"超高压"
            'i': '专业术语',     # 如"大陆漂移"
            'j': '简称',        # 如"深部"
            'l': '习用语',      # 如"几何学"
            'z': '状态词',      # 如"高温高压"
            'f': '方位词'       # 如"南缘"
        }

        # 英文词性到类别的映射
        self.en_pos_mapping = {
            'Content': ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'],
            'Function': ['ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'PART', 'PRON', 'SCONJ']
        }

        # 需要识别的实体类型（spaCy标准）
        self.entity_types = {
            'PERSON', 'NORP', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
            'EVENT', 'WORK', 'LAW', 'LANGUAGE'
        }
        self.ja_font_path = r"C:\Users\86157\Desktop\CLAP_Open\clap\src\CLAP\ipaex\ipaexg.ttf"
        self.ja_stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し',
            'も', 'ない', 'ます', 'です', 'だ', 'する', 'から', 'など'
        }
        
        # 日文词性映射
        self.ja_pos_mapping = {
            'Content': [
                '名詞', '動詞', '形容詞', '副詞', '形状詞',
                '感動詞', '接続詞', '代名詞', '連体詞', 'フィラー'
            ],
            'Function': [
                '助詞', '助動詞', '補助記号', '記号', 
                'フィラー', 'その他', '接頭辞', '接尾辞'
            ]
        }
        self.nlp_en = None
        model_paths = [
            r"C:\Users\86157\Desktop\env\en_core_web_md\en_core_web_md-3.8.0",  # 直接目录
            "en_core_web_md",  # 标准名称
            r"C:\Users\86157\anaconda3\envs\chat\Lib\site-packages\en_core_web_md"  # 可能的安装路径
        ]

        import site
        package_path = site.getsitepackages()[0]
        model_paths = [
            os.path.join(package_path, "en_core_web_md"),  # 标准安装路径
            "en_core_web_md"  # 逻辑名称
        ]
        
        # 添加调试信息
        print(f"🕵️ 正在搜索模型路径：")
        for path in model_paths:
            print(f" - {path} ({'存在' if os.path.exists(path) else '不存在'})")
        for path in model_paths:
            try:
                self.nlp_en = spacy.load(path)
                print(f"✅ 成功加载英文模型：{path}")
                break
            except Exception as e:
                print(f"⛔ 尝试路径 {path} 失败：{str(e)}")
        
        # 最终验证
        if not self.nlp_en:
            self._show_model_error_dialog()
        self.show()

    def _set_error_style(self):
        self.preview_info_label.setStyleSheet("""
            QLabel#doc_info {
                color: #dc3545;
                background: #fff5f5;
                border: 1px solid #fed7d7;
                padding: 12px;
                margin: 8px;
                border-radius: 4px;
            }
        """)

    def _show_model_error_dialog(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("模型加载失败")
        msg.setText("无法加载英文模型，请选择处理方式：")
        
        download_btn = msg.addButton("自动下载模型", QMessageBox.ActionRole)
        manual_btn = msg.addButton("手动指定路径", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("取消", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == download_btn:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        elif msg.clickedButton() == manual_btn:
            path = QFileDialog.getExistingDirectory(self, "选择模型目录")
            if path:
                try:
                    self.nlp_en = spacy.load(path)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    def on_new_text(self, text):
        # 在这里处理接收到的新文本，比如更新UI
        print(f"Received new text: {text}")
        # 假设你有一个 QTextEdit 或类似的部件来显示文本
        self.text_edit.append(text)

    #ui
    def init_ui(self):
        self.light_stylesheet = """
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei';
                color: #333333;
                background: #f8f9fa;
            }
            
            /* 工具栏美化 */
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border-bottom: 1px solid #dee2e6;
                spacing: 8px;
                padding: 4px;
            }
            
            /* 按钮样式 */
            QPushButton {
                background: rgba(0, 123, 255, 0.1); /* 半透明蓝色背景 */;
                color: orange;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 80px;
                border: 1px solid #007bff; /* 蓝色边框 */
            }
            
            QPushButton:hover {
                background:  rgba(0, 123, 255, 0.2);
            }
            
            QPushButton:pressed {
                background: rgba(0, 123, 255, 0.3);
            }
            
            /* 输入框美化 */
            QTextEdit, QTextBrowser {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background: white;
                selection-background-color: #b3d7ff;
            }
            
            /* 下拉框样式 */
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 2px 20px 2px 6px;
                min-width: 100px;
                background: white;
            }
            
            /* 文件列表样式 */
            QListWidget {
                background: #f8f9fa;
                alternate-background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            
            /* 标签样式 */
            QLabel[objectName^="doc_"] {
                font-size: 12px;
                color: #6c757d;
                padding: 4px;
            }
            
            /* 分割线样式 */
            QSplitter::handle {
                background: #dee2e6;
                width: 4px;
                margin: 2px;
            }
        """
        
        self.dark_stylesheet = """
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei';
                color: #ffffff;
                background: #2d2d2d;
            }
            
            /* 工具栏暗色版 */
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #303030);
                border-bottom: 1px solid #252525;
            }
            
            /* 按钮暗色样式 */
            QPushButton {
                background: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background: #4a4a4a;
            }
            
            QPushButton:pressed {
                background: #2a6496;
            }
            
            /* 输入框暗色 */
            QTextEdit, QTextBrowser {
                border: 1px solid #454545;
                border-radius: 4px;
                padding: 8px;
                background: #353535;
                color: #ffffff;
                selection-background-color: #454545;
            }
            
            /* 下拉框暗色 */
            QComboBox {
                border: 1px solid #454545;
                border-radius: 4px;
                padding: 2px 20px 2px 6px;
                min-width: 100px;
                background: #353535;
                color: #e0e0e0;
            }
            
            /* 文件列表暗色 */
            QListWidget {
                background: #303030;
                alternate-background-color: #252525;
                border: 1px solid #353535;
                border-radius: 4px;
                color: #ffffff;  # 列表文字白色
            }
            
            /* 标签样式 */
            QLabel[objectName^="doc_"] {
                font-size: 12px;
                color: #ffffff;
                padding: 4px;
            }
            
            /* 分割线样式 */
            QSplitter::handle {
                background: #252525;
                width: 4px;
                margin: 2px;
            }
        """
        
        # 初始化时使用light主题
        self.setStyleSheet(self.light_stylesheet)
        # 创建主窗口部件
        self.main_frame = QWidget()
        # 创建工具栏
        self.toolbar = QToolBar()   
        # 设置工具栏的文本大小
        self.toolbar.setStyleSheet("font-size: 12px")        
        # 将工具栏添加到主窗口
        self.addToolBar(self.toolbar)  
        # 创建翻译器
        self.translator = QTranslator(self)
        # 创建工具栏中的各个动作
        self.new_action = QAction('New Chat', self)
        self.open_action = QAction('Open Chat', self)
        self.save_action = QAction('Save Chat', self)
        self.export_action = QAction('To Markdown', self)          
        # 创建文本编辑框
        self.input_text_edit = QTextEdit()
        self.output_text_edit = QTextEdit()  
        # 创建文件查看器
        self.file_viewer = QWebEngineView()
        self.file_viewer.setSizePolicy(
            QSizePolicy.Expanding,  # 水平策略
            QSizePolicy.Expanding   # 垂直策略
        )
        self.file_viewer.setMinimumHeight(200)  
        # 创建文件列表部件
        self.file_list_widget = QListWidget()
        # 启用插件和PDF查看器
        self.file_viewer.settings().setAttribute(self.file_viewer.settings().WebAttribute.PluginsEnabled, True)
        self.file_viewer.settings().setAttribute(self.file_viewer.settings().WebAttribute.PdfViewerEnabled, True)
        # self.file_viewer.load(QUrl.fromLocalFile(current_directory + "/location.jpg"))
        self.file_viewer.setUrl(QUrl())
        self.text_browser = QTextBrowser() 
        self.text_browser.setAttribute(Qt.WA_TranslucentBackground)
        self.text_browser.setAttribute(Qt.WA_NoSystemBackground)
        self.text_browser.viewport().setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.text_browser.setFont(QFont("Consolas", 10))
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                font-size: 10pt;
                line-height: 1.5;
                border: none;
            }
            QScrollBar::vertical {
                width: 10px;
            }
            QScrollBar::handle:vertical {
                min-height: 20px;
            }
        """)
        # 禁用自动换行（保持原始换行结构）
        self.text_browser.document().setDefaultStyleSheet("""
            @keyframes char-fade {
                0% { opacity: 0; transform: translateY(2px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            .char-fade {
                animation: char-fade 0.1s ease-in;
                display: inline-block;
            }
            pre {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 4px;
                border: 1px solid #dee2e6;
                margin: 8px 0;
            }
            code {
                font-family: 'Consolas', 'Courier New';
                background: #fff5f5;
                padding: 2px 4px;
                border-radius: 2px;
            }
            blockquote {
                border-left: 3px solid #4a90e2;
                margin: 4px 0;
                padding-left: 8px;
                color: #6c757d;
            }
        """)
        self.import_button = QPushButton("Import\nCtrl+I")       
        self.send_button = QPushButton("Send\nCtrl+Enter")                
        self.role_label = QLabel("Role", self)
        self.role_selector = QComboBox(self)
        self.model_label = QLabel("Model", self)
        self.model_selector = QComboBox(self)               
        self.memory_label = QLabel("Memory", self)
        self.memory_selector = QComboBox(self)
        self.language_label = QLabel("Language", self)
        self.language_selector = QComboBox(self)
        self.mode_label = QLabel("模式：独立分析")# 在工具栏添加模式指示器
        self.toolbar.addWidget(self.mode_label)                
        self.toolbar.addAction(self.new_action)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addAction(self.export_action)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.model_label)
        self.toolbar.addWidget(self.model_selector)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.role_label)
        self.toolbar.addWidget(self.role_selector)     
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.memory_label)
        self.toolbar.addWidget(self.memory_selector)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.language_label)
        self.toolbar.addWidget(self.language_selector)    

        # 在工具栏中添加一个New action
        self.new_action.setShortcut('Ctrl+N')  # 设置快捷键为Ctrl+N
        self.new_action.triggered.connect(self.newChat)         
        # 在工具栏中添加一个Open action
        self.open_action.setShortcut('Ctrl+O')  # 设置快捷键为Ctrl+O
        self.open_action.triggered.connect(self.openChat)         
        # 在工具栏中添加一个Save action
        self.save_action.setShortcut('Ctrl+S') # 设置快捷键为Ctrl+S
        self.save_action.triggered.connect(self.saveChat)              
        # 在工具栏中添加一个Export action
        self.export_action.setShortcut('Ctrl+E') # 设置快捷键为Ctrl+E
        self.export_action.triggered.connect(self.exportMarkdown)  

        self.word_freq_action = QAction('📊 词频分析', self)
        self.word_freq_action.triggered.connect(self.show_word_frequency)
        self.toolbar.addAction(self.word_freq_action)
       # 添加清除历史按钮
        self.clear_history_action = QAction('🗑️ 清除历史', self)  # 添加图标
        self.clear_history_action.triggered.connect(self.clear_qa_history)
        self.toolbar.addAction(self.clear_history_action)  # 确保添加到工具栏

        self.export_history_action = QAction("📤 导出历史", self)
        self.export_history_action.triggered.connect(self.export_history)
        self.toolbar.addAction(self.export_history_action)

        roles = ['user', 'system', 'assistant']
        self.role_selector.addItems(roles)
        # 在工具栏添加日期选择控件
        self.date_filter = QComboBox()
        self.date_filter.addItems([
            "全部历史", 
            "最近7天", 
            "最近30天",
            "自定义范围"
        ])
        self.toolbar.addWidget(QLabel("时间筛选："))
        self.toolbar.addWidget(self.date_filter)

        self.theme_action = QAction('🌓 切换主题', self)
        self.theme_action.triggered.connect(self.toggle_theme)
        self.toolbar.addAction(self.theme_action)

        # 添加主题状态变量
        self.is_dark_theme = False
        
        self.multi_doc_btn = QPushButton("🔗 关联文档")
        self.multi_doc_btn.clicked.connect(self.link_documents)
        self.toolbar.addWidget(self.multi_doc_btn)

        self.link_indicator = QLabel("已关联0篇文档")
        self.toolbar.addWidget(self.link_indicator)

        # 添加模式切换按钮
        self.mode_switch = QPushButton("切换到多文档模式")
        self.mode_switch.setCheckable(True)
        self.mode_switch.clicked.connect(self.toggle_mode)
        self.toolbar.addWidget(self.mode_switch)

        memory_list = ['All', 'Input']
        self.memory_selector.addItems(memory_list)

        data = ollama.list()
        names = [model['model'] for model in data['models']]
        names.sort()
        self.model_selector.addItems(names)

        self.language_selector.currentTextChanged.connect(self.setLanguage)
        
        # 创建一个水平布局并添加表格视图和画布
        self.base_layout = QVBoxLayout()
        self.lower_layout = QHBoxLayout()
        self.upper_layout = QHBoxLayout()
        self.qm_files = [file for file in os.listdir()  if file.endswith('.qm')]
        # print(self.qm_files)
        self.language_selector.addItems(self.qm_files)
        # 创建一个新的字体对象
        font = QFont()
        font.setPointSize(12)
        # 设置字体
        self.input_text_edit.setFont(font)        
        self.input_text_edit.setAcceptDrops(True)
        self.input_text_edit.dragEnterEvent = self.dragEnterEvent
        self.input_text_edit.dropEvent = self.dropEvent

        self.output_text_edit.setFont(font)
        self.text_browser.setFont(font)

        # 创建一个QPushButton实例

        self.import_button.setShortcut('Ctrl+I')
        self.import_button.clicked.connect(self.importFile)
        self.import_button.setStyleSheet("font-size: 14px")
 
        self.send_button.setShortcut('Ctrl+Return') 
        self.send_button.clicked.connect(self.sendMessage)
        self.send_button.setStyleSheet("font-size: 14px")
        
        # 将文本编辑器和按钮添加到布局中
        # upper_layout.addWidget(self.output_text_edit)
        # 添加加载动画
        self.loading_gif = QMovie("loading.gif")
        self.loading_label = QLabel()
        self.loading_label.setMovie(self.loading_gif)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setVisible(False)  # 默认隐藏
        self.upper_layout.addWidget(self.loading_label)
        # self.upper_layout.addWidget(self.text_browser)

         # 加载模型列表
        data = ollama.list()
        names = [model['model'] for model in data['models']]
        names.sort()
        self.model_selector.addItems(names)

        # 设置默认模型
        if names:
            self.model_selector.setCurrentIndex(0)
            self.model = names[0]  # ✅ 初始化默认模型

        # 文件列表部件的信号连接
        self.file_list_widget.itemClicked.connect(self.load_pdf)

        self.preview_layout = QVBoxLayout()  # 初始化预览布局
        self.preview_info_label = QLabel()
        self.preview_layout.addWidget(self.preview_info_label)  # 先添加空标签占位

        # 调整布局比例和尺寸策略
        self._setup_layout()

        # 在工具栏添加文档计数
        self.doc_count_label = QLabel("已加载文档: 0")
        self.toolbar.addWidget(self.doc_count_label)

    @Slot(str)
    def _show_document_info(self, path: str):
        # 确保标签存在
        if not hasattr(self, 'preview_info_label'):
            self.preview_info_label = QLabel()
            self.preview_info_label.setObjectName("doc_info")
            self.preview_layout.addWidget(self.preview_info_label)
        
        # 设置基础样式
        self.preview_info_label.setStyleSheet("""
            QLabel#doc_info {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 12px;
                margin: 8px;
                min-height: 120px;
            }
        """)
        
        try:
            # 有效性检查
            if not self.doc_manager.load_success:
                self._set_error_view("⏳ 文档加载中...")
                return
                
            if not path or not os.path.isfile(path):
                self._set_error_view("⚠️ 文档路径无效")
                return
            
            # 先显示加载状态
            self.preview_info_label.setText("🔄 正在加载文档属性...")
            self.preview_info_label.setStyleSheet("color: #666;")
            QApplication.processEvents()  # 强制刷新界面
            
            # 获取文档信息
            doc_name = os.path.basename(path)
            file_size = os.path.getsize(path)
            mode = "多文档" if self.doc_manager.current_mode == "collection" else "独立"
            
            # 获取向量库信息
            try:
                collection = self.doc_manager.current_collection._collection
                chunk_count = collection.count() if collection else 0
                
                dimension = "N/A"
                if collection and chunk_count > 0:
                    peek_data = collection.peek()
                    if peek_data and 'embeddings' in peek_data:
                        embeddings = np.array(peek_data['embeddings'])
                        if embeddings.size > 0:
                            dimension = embeddings.shape[1]
            except Exception as e:
                chunk_count = "N/A"
                dimension = f"错误: {str(e)[:30]}"
            
            # 构建信息文本
            info_text = (
                f"📄 文档名称: {doc_name}\n"
                f"📦 文件大小: {file_size/1024:.2f} KB\n"
                f"📂 分析模式: {mode}模式\n"
                f"🔍 索引段落: {chunk_count}\n"
                f"🧮 向量维度: {dimension}\n"
                f"⏱ 更新时间: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            self.preview_info_label.setText(info_text)
            
        except Exception as e:
            self._set_error_view(f"❌ 属性加载失败: {str(e)}")
        # 文档名称处理
        max_name_length = 20  # 最大显示字符数
        doc_name = os.path.basename(path)
        display_name = (doc_name[:max_name_length] + '...') if len(doc_name) > max_name_length else doc_name
        
        # 设置提示文本
        self.preview_info_label.setToolTip(f"完整路径：{path}")  # 鼠标悬停显示全路径
        
        # 在信息文本中使用处理后的名称
        info_text = (
            f"📄 文档名称: {display_name}\n"  # 使用截断后的名称
            # ...其他信息行...
        )
        
        # 设置固定宽度
        self.preview_info_label.setMaximumWidth(300)  # 根据布局调整
        # 双击文档名称显示完整路径
        self.preview_info_label.mouseDoubleClickEvent = lambda event: (
            QMessageBox.information(
                self,
                "文档路径",
                f"完整路径：{self.current_file}"
            )
        )
        self.preview_info_label.setWordWrap(True)     # 启用自动换行

    def _set_error_view(self, message):
        """显示错误信息视图"""
        self.preview_info_label.setText(message)
        self.preview_info_label.setStyleSheet("""
            QLabel#doc_info {
                color: #dc3545;
                background: #fff5f5;
                border: 1px solid #fed7d7;
            }
        """)

    def _setup_layout(self):
        """配置主界面布局"""
        # 获取窗口高度
        window_height = self.height()
        # PDF预览区动态高度（占窗口80%）
        self.file_viewer.setMinimumHeight(int(window_height * 0.8))
        # 文档属性区自适应高度
        if not hasattr(self, 'preview_info_label'):
            self.preview_info_label = QLabel()
            self.preview_info_label.setObjectName("doc_info")
            self.preview_info_label.raise_()  # 确保信息标签在最上层
            self.preview_info_label.setStyleSheet("""
                background-color: rgba(255,255,255,0.9);
                border: 1px solid #ddd;
                border-radius: 2px;
            """)  # 添加背景防止被覆盖
        # 创建主分割器（左侧文件区，右侧内容区）
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧文件区域 (25%)
        file_splitter = QSplitter(Qt.Vertical)
        file_splitter.addWidget(self.file_list_widget)
        file_splitter.addWidget(self.import_button)
        file_splitter.setSizes([300, 100])  # 列表区域占3/4，按钮占1/4
        main_splitter.addWidget(file_splitter)
        
        # 右侧内容区域 (75%)
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.setStretchFactor(0, 2)  # 上部预览区伸缩因子
        content_splitter.setStretchFactor(1, 1)  # 下部结果展示区伸缩因子

        # 上部预览区 (40%)
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_main_layout = QVBoxLayout()
        preview_main_layout.setContentsMargins(0, 0, 0, 0)  # 移除外边距
        preview_main_layout.setSpacing(0)  # 移除间距
        # 创建统一标题栏
        title_bar = QWidget()
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)  # 减少外边距
        title_layout.setSpacing(5)  # 移除间距
        
        # 左侧标题（文档预览）
        preview_title = QLabel("📄 文档预览")
        preview_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #333;
                padding: 5px;
                margin-right: 10px;
            }
        """)
    
        # 右侧标题（文档属性）
        info_title = QLabel("📋 文档属性") 
        info_title.setStyleSheet(preview_title.styleSheet())
        
        # 添加到标题栏
        title_layout.addWidget(preview_title)
        title_layout.addStretch(1)  # 中间弹簧
        title_layout.addWidget(info_title)
        
        title_bar.setLayout(title_layout)
        
        # 添加标题栏到主布局
        preview_main_layout.addWidget(title_bar)
        
        # 内容区域（PDF + 属性）
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)  # 移除外边距
        content_layout.setSpacing(0)  # 移除间距
        
        # PDF预览部分（左侧60%）
        pdf_container = QWidget()
        pdf_layout = QVBoxLayout()
        pdf_layout.setContentsMargins(0, 0, 0, 0)  # 移除外边距
        pdf_layout.setSpacing(0)  # 移除间距
        pdf_layout.addWidget(self.file_viewer)
        pdf_container.setLayout(pdf_layout)
        
        # 文档属性部分（右侧40%）
        info_container = QWidget()
        info_container.setStyleSheet("""
            QWidget {
                min-width: 150px;
                max-width: 200px;
                padding: 5px;
            }
        """)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)  # 移除外边距
        info_layout.setSpacing(0)  # 移除间距
        info_layout.addWidget(self.preview_info_label)
        info_container.setLayout(info_layout)
    
        content_layout.addWidget(pdf_container, 8)  # 4:1比例
        info_container.setMinimumWidth(150)  # 最小宽度限制
        content_layout.addWidget(info_container, 1)  # 保持1:4的比例
        
        preview_main_layout.addLayout(content_layout)
        preview_frame.setLayout(preview_main_layout)
        
        # 下部对话区 (60%)
        chat_frame = QFrame()
        chat_frame.setFrameShape(QFrame.StyledPanel)
        chat_layout = QVBoxLayout()
        chat_title = QLabel("💬 对话记录")
        chat_title.setStyleSheet(preview_title.styleSheet())  # 使用相同的样式
    
        # 添加对话记录标题到布局
        chat_layout.addWidget(chat_title)
        chat_layout.addWidget(self.text_browser)
        chat_frame.setLayout(chat_layout)
        
        content_splitter.addWidget(preview_frame)
        content_splitter.addWidget(chat_frame)
        content_splitter.setSizes([400, 600])  # 预览:对话 = 4:6
        
        main_splitter.addWidget(content_splitter)
        
        # 设置整体比例 (文件区:内容区 = 25%:75%)
        main_splitter.setSizes([250, 750])
        
        # 底部输入区
        input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_text_edit, 5)  # 输入框占5份
        input_layout.addWidget(self.send_button, 1)      # 按钮占1份
        input_frame.setLayout(input_layout)
        # 按钮垂直布局
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        # 添加操作按钮（修正处：正确定义clear_button）
        self.pause_button = QPushButton("Pause\nCtrl+P")
        self.pause_button.setShortcut('Ctrl+P')
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet("font-size: 14px")
        
        # 正确定义clear_button（添加此行）
        self.clear_button = QPushButton("Clear\nCtrl+Shift+L")
        self.clear_button.setShortcut('Ctrl+Shift+L')
        self.clear_button.clicked.connect(self.clear_current_conversation)
        self.clear_button.setStyleSheet("font-size: 14px")
        
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.clear_button)  # 使用self.clear_button
        button_layout.addWidget(self.send_button)

        input_layout.addWidget(self.input_text_edit, 5)
        input_layout.addLayout(button_layout, 1)
        
        input_frame.setLayout(input_layout)
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter, 8)         # 主内容区占80%
        main_layout.addWidget(input_frame, 2)           # 输入区占20%
        
        main_layout.addWidget(self.loading_label, alignment=Qt.AlignCenter)

        # 尺寸策略优化
        self.file_list_widget.setMinimumWidth(200)
        self.file_viewer.setMinimumSize(400, 300)
        self.text_browser.setMinimumHeight(200)
        self.input_text_edit.setMinimumHeight(80)
        
        # 字体和边距调整
        font = QFont("Microsoft YaHei", 10)
        self.file_list_widget.setFont(font)
        self.text_browser.setFont(font)
        self.input_text_edit.setFont(font)
        
        main_layout.setContentsMargins(5, 5, 5, 5)      # 整体边距
        main_layout.setSpacing(3)                       # 部件间距
        preview_main_layout.setContentsMargins(2, 2, 2, 2)  # 边距从5px减少到2px
        content_layout.setSpacing(5)  # 组件间距从20px减少到5px

        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 加载动画定位
        self.loading_label.setFixedSize(100, 100)
        self.loading_label.move(
            self.width()//2 - 50, 
            self.height()//2 - 50
        )
        # 更新文件列表
        self.update_file_list()

    def debug_word_span(self, word):
        """命令行调试某个词的段落分布"""
        if not hasattr(self, 'word_para_counts'):
            print("请先进行词频分析")
            return
        
        print(f"调试词汇: {word}")
        print(f"统计段落数: {self.word_para_counts.get(word, 0)}")
        
        raw_text = self.doc_manager.get_raw_text(self.get_current_file_path())
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
        
        matches = []
        for idx, para in enumerate(paragraphs):
            if word in para:
                print(f"📖 段落 {idx+1}: {para[:80]}...")
                matches.append(idx+1)
        
        print(f"🔍 实际匹配段落数: {len(matches)}")
        print(f"📊 统计差异: {self.word_para_counts.get(word,0)} vs {len(matches)}")
    
    def analyze_relations(self, text, is_chinese):
        """概念关系分析"""
        if is_chinese:
            return self._analyze_chinese_relations(text)
        else:
            return self._analyze_english_relations(text)
    
    def _analyze_english_relations(self, text):
        """英文关系分析"""
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        
        relations = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "dobj"):
                    subj = token.head.text
                    rel = token.dep_
                    obj = token.text
                    relations.append((subj, rel, obj))
        return relations
    
    def _analyze_chinese_relations(self, text):
        """中文关系分析"""
        words = pseg.cut(text)
        relations = []
        buffer = []
        
        for word, flag in words:
            if flag.startswith('n'):
                if len(buffer) >= 2:
                    subj = buffer[-2]
                    rel = buffer[-1]
                    obj = word
                    relations.append((subj, rel, obj))
                buffer.append(word)
            elif flag.startswith('v'):
                if buffer:
                    subj = buffer[-1]
                    rel = word
                    obj = ''
                    relations.append((subj, rel, obj))
        return relations
    
    def show_word_frequency(self):
        """显示词频分析窗口"""
        # 显示加载提示
        progress = QProgressDialog("正在分析文档内容...", None, 0, 0, self)
        progress.setWindowTitle("请稍候")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            if self.doc_manager.current_mode == "collection":
                QMessageBox.warning(self, "模式错误", "请切换到单文档模式进行词频分析")
                return
                    
            current_path = self.get_current_file_path()
            if not current_path:
                QMessageBox.warning(self, "文档错误", "请先选择要分析的文档")
                return
                
            raw_text = self.doc_manager.get_raw_text(current_path)
            if not raw_text:
                QMessageBox.critical(self, "内容错误", "无法获取文档文本内容")
                return
                
            # 先判断语言类型
            is_chinese = self.is_chinese(raw_text)
            is_japanese = self.is_japanese(raw_text)
            
            # 设置停用词（这里应该使用实例变量）
            if is_chinese:
                stop_words = self.zh_stop_words
            elif is_japanese:
                stop_words = self.ja_stop_words
            else:
                stop_words = self.en_stop_words
            
            # 分词逻辑
            words = []
            if is_chinese:
                # 中文分词
                words = jieba.lcut(raw_text)
                words = [w for w in words if len(w) > 1 and w not in stop_words and not w.isdigit()]
            elif is_japanese:
                # 日文分词
                from janome.tokenizer import Tokenizer
                t = Tokenizer()
                tokens = t.tokenize(raw_text)
                words = [token.base_form for token in tokens 
                        if token.part_of_speech.split(',')[0] not in ['助詞', '助動詞', '記号']
                        and len(token.base_form) > 1]
            else:
                from nltk.tokenize import word_tokenize
                # 英文分词
                words = word_tokenize(raw_text)
                words = [w.lower() for w in words if w.isalpha() and len(w) > 2 and not w.isdigit()]
                words = [w for w in words if w not in stop_words]  
            if not words:
                QMessageBox.warning(self, "分析结果", "未找到有效词汇")
                return
                
            # 统计词频
            counter = Counter(words)
            top_words = counter.most_common(20)

            # 段落统计逻辑
            paragraphs = [p for p in raw_text.split('\n') if len(p.strip()) > 0]
            self.word_para_counts = defaultdict(int)
            
            for para_idx, para in enumerate(paragraphs):
                # 使用与主分词相同的方法处理段落
                if is_chinese:
                    para_words = jieba.lcut(para)
                elif is_japanese:
                    t = Tokenizer()
                    para_words = [token.base_form for token in t.tokenize(para)]
                else:
                    para_words = word_tokenize(para)
                
                unique_words = set(w for w in para_words if w not in stop_words)
                for word in unique_words:
                    self.word_para_counts[word] += 1

            # 将段落统计合并到词频数据
            top_words = [
                (word, count, self.word_para_counts.get(word, 0))  
                for (word, count) in counter.most_common(20)
            ]

            # 生成验证报告
            validation_report = []
            for word, count, span in top_words[:5]:  # 检查前5个高频词
                actual_span = 0
                para_details = []
                
                for para_idx, para in enumerate(paragraphs):
                    if word in para:  # 实际匹配检查
                        actual_span += 1
                        para_details.append(f"段落 {para_idx+1}: {para[:100]}...")
                
                validation_report.append({
                    'word': word,
                    'expected': span,
                    'actual': actual_span,
                    'match': span == actual_span,
                    'paragraphs': para_details
                })
            
            # 保存验证报告
            self.validation_data = validation_report
            print("【验证报告】", validation_report)

            # 显示分析窗口
            self.word_freq_window = WordFrequencyWindow(self)
            analysis_result = self.analyze_text(raw_text, is_chinese)
            pos_stats = analysis_result.get('pos', {})
            entities = analysis_result.get('entities', {})
            relations = self.analyze_relations(raw_text, is_chinese)

            print(f"【分析结果】词性统计: {pos_stats}")
            print(f"【分析结果】实体统计: {entities}")
            print(f"【分析结果】概念关系: {relations}")

            self.word_freq_window.plot_bar(top_words, is_chinese)
            self.word_freq_window.plot_wordcloud(raw_text, is_chinese)
            self.word_freq_window.plot_pos(pos_stats, is_chinese)
            self.word_freq_window.plot_entities(entities, is_chinese)
            self.word_freq_window.plot_relations(relations, is_chinese)
            self.word_freq_window.show()

            # 保存词频数据和语言状态
            self.word_counts = top_words  # 保存词频数据
            self.is_chinese_flag = is_chinese  # 保存语言状态
            self.pos_stats = analysis_result.get('pos', {})
            self.entities = analysis_result.get('entities', {})
            self.relations = relations

        except Exception as e:
            QMessageBox.critical(self, "分析错误", f"发生未知错误：{str(e)}")
        finally:
            progress.close()

    def is_chinese(self, text, threshold=0.3):
        """自动判断文本语言"""
        # 检测日文假名
        ja_kana = len(re.findall(r'[\u3040-\u30FF]', text))
         # 中文字符（排除日文汉字）
        chn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = max(len(text), 1)  # 防止除零
        return (chn_chars / total_chars) > threshold and ja_kana < chn_chars * 0.2
    
    def is_japanese(self, text, threshold=0.2):
        """判断是否为日文文本"""
        # 检测日文假名
        ja_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u3000-\u303F]'
        ja_chars = len(re.findall(ja_pattern, text))
        total_chars = max(len(text), 1)
        return (ja_chars / total_chars) > threshold 
    

    def analyze_text(self, text, is_chinese):
        pos_stats = defaultdict(int)
        entities = defaultdict(int)
        concepts = []
        keywords = []
        
        try:
            # 精确语言判断
            is_japanese = self.is_japanese(text)
            is_chinese = self.is_chinese(text) if not is_japanese else False

            # 日文处理分支
            if is_japanese:
                from janome.tokenizer import Tokenizer
                t = Tokenizer()
                tokens = list(t.tokenize(text))

                # 打印前10个token的词性
                print("【日语词性诊断】样例词性:", [
                    (token.surface, token.part_of_speech) 
                    for token in tokens[:10]
                ])

                # 词性统计
                for token in tokens:
                    # 获取词性的第一层级（如 "名詞,一般,*,*" -> "名詞"）
                    primary_pos = token.part_of_speech.split(',')[0]
                    matched = False
    
                    # 优先检查Content类
                    for pos_type in self.ja_pos_mapping['Content']:
                        if primary_pos.startswith(pos_type):
                            pos_stats['Content'] += 1
                            matched = True
                            break
                            
                    if not matched:
                        # 检查Function类
                        for pos_type in self.ja_pos_mapping['Function']:
                            if primary_pos.startswith(pos_type):
                                pos_stats['Function'] += 1
                                matched = True
                                break
                                
                    # 未匹配的归类到Other
                    if not matched:
                        pos_stats['Other'] += 1
                
                # 扩展实体识别类型
                entity_rules = {
                    '名詞,固有名詞,人名': '人物',
                    '名詞,固有名詞,組織': '組織',
                    '名詞,固有名詞,地域': '地域',
                    '名詞,固有名詞,一般': '固有名詞'
                }

                for token in tokens:
                    pos = token.part_of_speech
                    # 精确匹配实体类型
                    for pattern, label in entity_rules.items():
                        if pos.startswith(pattern):
                            entities[label] += 1
                            break
                        # 通用名词统计
                        elif pos.startswith('名詞,一般'):
                            entities['一般名詞'] += 1
                
                print(f"【日语实体诊断】识别结果: {dict(entities)}")
                
                print(f"【日语词性诊断】最终统计: {pos_stats}")
                return {
                    'pos': dict(pos_stats),
                    'entities': dict(entities),
                    'concepts': [],
                    'keywords': []
                }    
            
            # 前置文本清洗
            chn_text, eng_text = self.clean_mixed_text(text)
            
            # 添加诊断日志（修正位置）
            print(f"【分词诊断】输入文本长度: {len(text)} 首50字: {text[:50]}")
            print(f"【分词诊断】清洗后中文部分: {chn_text[:100]}..." if chn_text else "无中文内容")
            print(f"【分词诊断】清洗后英文部分: {eng_text[:100]}..." if eng_text else "无英文内容")

            
            # 分词处理
            words = []
            if is_chinese:
                words = self.hybrid_segmentation(chn_text, eng_text)
            else:
                words = self.english_analysis(text)
                if not words:
                    raise ValueError("英文分析未返回有效结果")

            # 确保 words 是一个列表
            if not isinstance(words, list):
                raise TypeError(f"Expected a list of tuples, but got {type(words).__name__}")

            # 打印 words 内容以便调试
            print(f"【分词诊断】分词结果前10项: {words[:10]}")

            # ========== 词性分析 ==========
            if is_chinese and self.ltp:
                seg, hidden = self.ltp.seg([chn_text])  # 使用清洗后的中文文本
                postag = self.ltp.postag(hidden)
                
                # 处理识别结果
                for sent_postag in postag:
                    for item in zip(seg[0], sent_postag):
                        word, flag = item
                        flag = flag.split('.')[0]  # 处理复合标签
                        
                        # 词性分类逻辑
                        if flag in self.zh_pos_mapping['实词']:
                            pos_stats['Content'] += 1
                        elif flag in self.zh_pos_mapping['虚词']:
                            pos_stats['Function'] += 1
                        elif flag == 'ENG':
                            pos_stats['Content'] += 1  # 英文术语视为内容词
                        
                        # 新词性发现机制
                        if flag not in self.zh_pos_mapping['实词'] | self.zh_pos_mapping['虚词'] | {'ENG'}:
                            explanation = self.zh_pos_explain.get(flag, "未知词性")
                            print(f"发现新词性: {flag} ({explanation}) | 词语: {word}")
                            if explanation in ["专业术语", "状态词"]:
                                pos_stats['Content'] += 1

                # 实体识别
                ner = self.ltp.ner(hidden)
                for sent_ner in ner:
                    for item in sent_ner:
                        ent_type = self.entity_mapping.get(item[1], "其他")
                        entities[ent_type] += 1
                        concepts.append(chn_text[item[0]:item[1]+1])

                # 概念提取
                concepts = [
                    word for word, flag in words 
                    if word and (flag.startswith(('n', 'v', 'a')) or flag == 'ENG')
                ]
            else:
                # 英文处理流程
                doc = self.nlp_en(text)
                for token in doc:
                    if token.pos_ in self.en_pos_mapping['Content']:
                        pos_stats['Content'] += 1
                    elif token.pos_ in self.en_pos_mapping['Function']:
                        pos_stats['Function'] += 1

                # 实体识别
                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        entities[self.entity_mapping.get(ent.label_, ent.label_)] += 1
                        concepts.append(ent.text)
            
            # 关键词提取
            clean_words = [self.normalize_word(word) for word, _ in words if word]
            keywords = self.extract_keywords(clean_words)

            # 打印 pos_stats 以便调试
            print(f"【分词诊断】词性统计: {dict(pos_stats)}")

            return {
                'pos': dict(pos_stats),
                'entities': dict(entities),
                'concepts': concepts,
                'keywords': keywords
            }
        
        except Exception as e:
            print(f"全局分析错误: {str(e)}")
            traceback.print_exc()
            return {}
        
    def clean_mixed_text(self, text):
        """混合文本深度清洗"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ("", "")  # 确保始终返回二元组
        try:
            # 基础清洗
            cleaned = re.sub(
                r'[^\u4e00-\u9fa5a-zA-Z0-9\s,\.\?!;:“”‘’（）《》—\-]',
                ' ',
                text
            )
            
            # 特殊格式处理
            patterns = [
                (r'http[s]?://\S+', ' '),
                (r'\b\d{4}年?代?\b', ' '),
                (r'©|®|™', ' '),
                (r'\s+', ' '),
                (r'([a-zA-Z])/([a-zA-Z])', r'\1\2')
            ]
            
            for pattern, replacement in patterns:
                cleaned = re.sub(pattern, replacement, cleaned)
            
            # 中英文分离
            chn_part = re.sub(r'[^\u4e00-\u9fa5]', ' ', cleaned)
            eng_part = ' '.join(re.findall(r'\b[a-zA-Z]{3,}\b', cleaned))
            
            return chn_part.strip(), eng_part.strip()
        except Exception as e:
            print(f"文本清洗异常: {str(e)}")
            return ("", "")  # 异常时返回安全值
        
    def english_analysis(self, text):
        """英文文本分析管道"""
        # 空值保护
        if not text.strip():
            return []
        
        try:
            # 使用spacy进行分词和词性标注
            doc = self.nlp_en(text)
            result = [
                (token.text, token.pos_) 
                for token in doc 
                if not token.is_stop and len(token.text) > 2
            ]
            # 打印结果以便调试
            print(f"【分词诊断】英文分词结果前10项: {result[:10]}")
            return result
        except Exception as e:
            print(f"英文分析失败: {str(e)}")
            return []

    def _init_english_model(self):
        """延迟加载英文模型"""
        try:
            self.nlp_en = spacy.load("en_core_web_md")
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("缺少英文语言包")
            msg.setInformativeText(
                "请执行以下命令安装：\npython -m spacy download en_core_web_sm"
            )
            msg.exec_()
            raise RuntimeError("英文模型未安装")

    def hybrid_segmentation(self, chn_text, eng_text):
        segments = []
        
        try:
            # 中文处理
            # 加载自定义词典
            script_dir = os.path.dirname(os.path.abspath(__file__))
            geology_dict_path = os.path.join(script_dir, 'geology_dict.txt')
            if not os.path.exists(geology_dict_path):
                raise FileNotFoundError(f"词典文件未找到: {geology_dict_path}")
            jieba.load_userdict(geology_dict_path)
            for element in pseg.cut(chn_text):
                # 添加类型断言
                word, flag = self.validate_segment(element)
                if not isinstance(word, str) or not isinstance(flag, str):
                    raise ValueError(f"非法分词结果 word类型: {type(word)} flag类型: {type(flag)}")
                    
                segments.append((word, flag))
            
            # 英文术语处理（添加长度校验）
            geo_terms = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b', eng_text)
            for term in geo_terms:
                if len(term) > 3:  # 过滤短词
                    normalized = term.replace('-', '_').strip()
                    segments.append((normalized, 'GEO_TERM'))
                    
            return segments  # 确保始终返回列表二元组
        
        except Exception as e:
            print(f"分词流程异常: {str(e)}")
            return []  # 返回空列表避免解包错误

    def validate_segment(self, element):
        try:
            if hasattr(element, 'word') and hasattr(element, 'flag'):
                return (element.word, element.flag)
            
            if isinstance(element, tuple) and len(element) >= 2:
                return (str(element[0]).strip(), str(element[1]).strip())
            
            if isinstance(element, str):
                return (element.strip(), 'x')
                
            return (str(element).strip(), 'x')
        except Exception as e:
            print(f"分词验证异常: {str(e)}")
            return ('', 'x')

    def normalize_word(self, word):
        """统一处理特殊符号"""
        return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5_]', '', word).lower()

    def extract_keywords(self, words):
        """关键词提取（带空值保护）"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                token_pattern=r'(?u)\b\w+\b',
                stop_words=list(self.en_stop_words | self.zh_stop_words)
            )
            tfidf = vectorizer.fit_transform([' '.join(words)])
            feature_names = vectorizer.get_feature_names_out().tolist()
            print(f"【分词诊断】关键词提取结果: {feature_names}")
            return feature_names
        except ValueError:
            return []
        
    
    def clear_qa_history(self):
        """清除当前文档的问答历史"""
        current_file = self.get_current_file_path()
        if not current_file:
            QMessageBox.warning(self, "错误", "请先选择要清除历史的文档")
            return
            
        # 获取文档基础名称用于显示
        base_name = os.path.basename(current_file)
        
        # 添加二次确认对话框
        confirm = QMessageBox.question(
            self,
            "确认清除",
            f"确定要清除文档《{base_name}》的所有对话历史吗？\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # 执行数据库删除
                cursor = self.doc_manager.conn.cursor()
                cursor.execute(
                    "DELETE FROM qa_history WHERE file_path=?",
                    (current_file,)
                )
                self.doc_manager.conn.commit()
                
                # 更新界面显示
                self._load_qa_history(current_file)
                QMessageBox.information(self, "成功", "历史记录已清除")
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "数据库错误",
                    f"清除历史失败：{str(e)}"
                )
    def export_history(self):
        """导出当前文档的历史记录"""
        current_file = self.get_current_file_path()
        if not current_file:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出历史记录",
            f"{os.path.basename(current_file)}_history.csv",
            "CSV Files (*.csv)"
        )
        
        if path:
            history = self.doc_manager.get_questions_answers(current_file)
            pd.DataFrame(history).to_csv(path, index=False)
            
    def _load_qa_history(self, file_path):
        """加载指定文件的问答历史"""
        try:
            if not file_path:
                return
                
            # 从文档管理器获取问答列表
            qa_list = self.doc_manager.get_questions_answers(file_path)
            
            # 清空当前显示
            self.text_browser.clear()
            
            # 格式化成带序号的列表
            history_text = ""
            for i, qa in enumerate(qa_list, 1):
                history_text += f"{i}. [Q] {qa['question']}\n   [A] {qa['answer']}\n\n"
                
            # 添加最后一条消息（如果有）
            if self.messages.get(file_path):
                last_msg = self.messages[file_path][-1]
                history_text += f"最新对话:\n{last_msg['content']}"
                
            # 更新显示
            self.text_browser.setText(history_text)
            
        except KeyError:
            print(f"尚未保存 {os.path.basename(file_path)} 的问答历史")
        except Exception as e:
            QMessageBox.warning(self, "历史加载错误", 
                f"无法加载问答历史:\n{str(e)}")
            
    def toggle_theme(self):
        if self.is_dark_theme:
            # 切换到浅色主题
            self.setStyleSheet(self.light_stylesheet)
            # 特殊控件样式重置
            self.text_browser.setStyleSheet("""
                QTextBrowser { background: white; color: black; }
                QScrollBar::handle { background: #888; }
            """)
        else:
            # 切换到深色主题
            self.setStyleSheet(self.dark_stylesheet)
            # 特殊控件样式调整
            self.text_browser.setStyleSheet("""
                QTextBrowser { background: #353535; color: #ffffff; }
                QScrollBar::handle { background: #666; }
            """)

        self.is_dark_theme = not self.is_dark_theme
        # 强制刷新所有控件样式
        self.style().polish(self)
            
    def link_documents(self):
        """文档关联对话框"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("选择关联文档（Ctrl多选）")
            dialog.resize(400, 300)  # 设置合适的大小
            
            layout = QVBoxLayout()
            
            # 文档列表
            list_widget = QListWidget()
            list_widget.setSelectionMode(QListWidget.MultiSelection)
            
            # 加载所有已加载文档
            for path in self.doc_manager.loaded_paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.UserRole, path)
                list_widget.addItem(item)
            
            # 按钮组
            btn_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                parent=dialog
            )
            btn_box.accepted.connect(dialog.accept)
            btn_box.rejected.connect(dialog.reject)
            
            # 布局管理
            layout.addWidget(QLabel("选择要关联分析的文档："))
            layout.addWidget(list_widget)
            layout.addWidget(btn_box)
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.Accepted:
                selected_paths = [item.data(Qt.UserRole) for item in list_widget.selectedItems()]
                self.active_links = selected_paths
                # 生成唯一会话ID
                if selected_paths:
                    sorted_paths = sorted(p for p in selected_paths if isinstance(p, str))
                    combined = ','.join(sorted_paths).encode('utf-8')
                    session_id = hashlib.md5(combined).hexdigest()
                    self.current_session = session_id
                else:
                    self.current_session = None
                self.link_indicator.setText(f"已关联{len(selected_paths)}篇文档")
                print(f"关联文档: {selected_paths}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "文档关联错误",
                f"无法完成文档关联:\n{str(e)}"
            )
            traceback.print_exc()

    def toggle_mode(self):
        if self.mode_switch.isChecked():
            self.doc_manager.current_mode = "collection"
            self.mode_label.setText("模式：多文档联合分析")
        else:
            self.doc_manager.current_mode = "single"
            self.mode_label.setText("模式：单文档分析")
            
        # 清空当前会话上下文
        self.current_session = None
            
    def toggle_pause(self):
    # 增加线程状态检查
        if not self.current_chat_thread or not self.current_chat_thread.isRunning():
            print("没有正在运行的线程")
            return
        
        # 添加调试日志
        print(f"当前暂停状态: {self.is_paused}")
        print(f"线程活跃状态: {self.current_chat_thread.isRunning()}")
        
        if self.is_paused:
            # 确保唤醒后重置状态
            self.current_chat_thread.resume()
            self.loading_gif.start()
            self.pause_button.setText("Pause\nCtrl+P")
            self.pause_button.setStyleSheet("")
        else:
            # 添加强制暂停保护
            self.current_chat_thread.pause()
            self.loading_gif.stop()
            self.pause_button.setText("继续\nCtrl+P") 
            self.pause_button.setStyleSheet("background-color: #ff9999;")

        self.is_paused = not self.is_paused

    def clear_current_conversation(self):
        if not self.get_current_file_path():
            self.text_browser.clear()
            self.input_text_edit.clear()
        """清空当前对话上下文"""
        # 终止正在运行的线程
        if hasattr(self, 'chat_thread') and self.chat_thread.isRunning():
            self.chat_thread.terminate()
        
        # 清空输入框
        self.input_text_edit.clear()
        
        # 保留显示的历史记录，仅清空当前会话的上下文
        current_file = self.get_current_file_path()
        if current_file and current_file in self.messages:
            self.messages[current_file] = []
        
        # 重置暂停状态
        self.is_paused = False
        self.pause_button.setText("Pause\nCtrl+P")
        
        # 停止加载动画
        self.loading_gif.stop()
        self.loading_label.hide()
    def update_file_list(self):
        pdf_files = [file for file in os.listdir('.') if file.endswith('.pdf')]
        self.file_list_widget.clear()
        for file in pdf_files:
            file_path = os.path.join('.', file)
            file_item = QListWidgetItem(file)
            file_item.setData(Qt.UserRole, file_path)  # 存储完整路径
            self.file_list_widget.addItem(file_item)

        # 默认选择第一个文档
        if self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            self.on_file_selected(self.file_list_widget.item(0))

    def load_pdf(self, item):
        file_path = item.data(Qt.UserRole)
        self.file_viewer.setUrl(QUrl.fromLocalFile(file_path))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.path = path
            self.file_viewer.load(QUrl.fromLocalFile(path))
            print('import file at', path)
    def setLanguage(self):                
        # 加载.qm文件
        self.translation = self.language_selector.currentText()
        # print(self.translation)
        self.translator.load(self.translation)
        QApplication.installTranslator(self.translator)

        self.text_labels={
                'title': QApplication.translate('Context', 'CLAP: Chat Local And Persistent. Based on Ollama, a Graphical User Interface for Local Large Language Model Conversations'),
                'new': QApplication.translate('Context', 'New Chat'),
                'open': QApplication.translate('Context', 'Open Chat'),
                'save': QApplication.translate('Context', 'Save Chat'),
                'export': QApplication.translate('Context', 'To Markdown'),
                'model': QApplication.translate('Context', 'Model'),
                'memory': QApplication.translate('Context', 'Memory'),
                'role': QApplication.translate('Context', 'Role'),
                'import': QApplication.translate('Context', 'Import')+'\nCtrl+I',
                'send': QApplication.translate('Context', 'Send')+'\nCtrl+Enter',
                'language': QApplication.translate('Context', 'Language'),
                'input_text': QApplication.translate('Context', 'Input'),
                'output_text': QApplication.translate('Context', 'Output'),
                'timestamp': QApplication.translate('Context', 'Timestamp')
            }   

        self.setWindowTitle(self.text_labels['title'])
        self.new_action.setText(self.text_labels['new'])
        self.open_action.setText(self.text_labels['open'])
        self.save_action.setText(self.text_labels['save'])
        self.export_action.setText(self.text_labels['export'])
        self.model_label.setText(self.text_labels['model'])
        self.memory_label.setText(self.text_labels['memory'])
        self.role_label.setText(self.text_labels['role'])
        self.import_button.setText(self.text_labels['import'])
        self.send_button.setText(self.text_labels['send'])
        self.language_label.setText(self.text_labels['language'])

    def resizeEvent(self, event):
        # 获取窗口的新大小
        new_width = event.size().width()
        new_height = event.size().height()
        # 保持PDF区域最小高度为窗口高度的40%
        min_pdf_height = int(self.height() * 0.4)
        self.file_viewer.setMinimumHeight(min_pdf_height)
        
        # 强制更新布局
        QApplication.processEvents()
        super().resizeEvent(event)

    # 在主窗口类中
    def start_chat(self,prompt,file_path):
        # 获取当前的日期和时间
        now = datetime.now()
        # 将日期和时间格式化为字符串
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        self.show_text += '\n\n'+ self.text_labels['model']+ ' '  + self.model + '\t' + self.text_labels['role'] + ' '  + self.role  +  '\t' + self.text_labels['timestamp'] + ' '  + timestamp + '\n' + self.text_labels['input_text']+ ' '   + ': ' + self.input_text + '\n' + self.text_labels['output_text']+ ' '  

        self.text_browser.setText(self.show_text) # 将文本添加到文本浏览器中
        self.chat_thread = ChatThread(
            prompt=prompt,  # 使用增强后的提示
            messages=self.messages[file_path], 
            document_manager=self.doc_manager, 
            model=self.model,
            paths=[self.path] if self.path else [] 
        )
        self.chat_thread.new_text.connect(self.update_text_browser)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.start()
    
    def on_chat_finished(self, final_response):
        # 线程清理
        if self.current_chat_thread:
            try:
                self.current_chat_thread.disconnect()  # 断开所有信号
            except:
                pass
            self.current_chat_thread.quit()  # 请求线程退出
            self.current_chat_thread.deleteLater()
            self.current_chat_thread = None
        print("输出已完成")
        # 确保响应有效性
        final_response = str(final_response).strip()
        if not final_response:
            print("收到空响应，跳过处理")
            return
        # 更新消息列表中的最后一个消息内容
        current_file_path = self.get_current_file_path()
        if current_file_path:
            session_id = f"multi_{hash(frozenset(self.active_links))}" if self.doc_manager.current_mode == "collection" else current_file_path
            self.doc_manager.update_conversation_context(
                session_id,
                self.input_text,
                final_response
            )
        # 有效性检查
        if (current_file_path and 
            os.path.exists(current_file_path) and 
            hasattr(self, 'input_text')):
            
            try:
                # 更新对话历史
                if self.messages.get(current_file_path):
                    last_msg = self.messages[current_file_path][-1]
                    if self.memory_selector.currentText() != 'Input':
                        last_msg['content'] += f"\n[模型回复]\n{final_response}"
                # 构建问答记录
                qa_record = {
                    'question': self.input_text,
                    'answer': final_response,
                    'timestamp': datetime.now().isoformat()
                }
            
                # 获取现有记录并追加
                existing = self.doc_manager.get_questions_answers(current_file_path)
                existing.append(qa_record)
                self.doc_manager.set_questions_answers(current_file_path, existing)
            
            except Exception as e:
                print(f"保存QA记录失败: {str(e)}")

            # 刷新显示
            self._load_qa_history(current_file_path) 
            
            # 清除输入框的内容
            self.input_text_edit.clear()
            
            # 停止加载动画
            self.loading_gif.stop()
            self.loading_label.hide()
            
            # 重置暂停状态
            self.is_paused = False
            self.pause_button.setText("Pause\nCtrl+P")

    def _should_auto_scroll(self):
        scrollbar = self.text_browser.verticalScrollBar()
        return scrollbar.value() + scrollbar.pageStep() >= scrollbar.maximum()

    def update_text_browser(self, text):

        self.text_browser.moveCursor(QTextCursor.End)
        self.text_browser.insertPlainText(text)
        
        # 1. 自动滚动控制（确保新内容可见）
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 2. 性能优化（避免频繁刷新）
        if len(text) > 100:  # 长文本分批处理
            QApplication.processEvents()
            
        # 3. 保留原始换行符处理
        self.text_browser.ensureCursorVisible()
        
        # 4. 颜色标记（可选）
        if "ERROR" in text:
            self.text_browser.setTextColor(QColor("#FF0000"))  # 错误信息红色
        else:
            self.text_browser.setTextColor(QColor("#000000"))  # 默认黑色
        
    

    
    def _smooth_scroll(self):
        """惯性滚动动画"""
        scrollbar = self.text_browser.verticalScrollBar()
        anim = QPropertyAnimation(scrollbar, b"value")
        anim.setDuration(300)
        anim.setEasingCurve(QEasingCurve.OutQuint)
        anim.setEndValue(scrollbar.maximum())
        anim.start()

    def _flush_updates(self):
        if self.pending_updates:
            combined = "".join(self.pending_updates)
            self._safe_insert_html(combined)
            self.pending_updates.clear()
        self.update_timer.stop()

    def _safe_insert_html(self, html):
        """线程安全的HTML插入方法"""
        try:
            if not self.style_initialized:
                self._inject_base_styles()
                self.style_initialized = True
            cursor = self.text_browser.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(html)
            
            # 自动滚动控制
            if self._should_auto_scroll():
                self.text_browser.moveCursor(QTextCursor.End)
                self.text_browser.ensureCursorVisible()
        except Exception as e:
            print(f"HTML插入失败: {str(e)}")
            self.text_browser.insertPlainText(html)  # 降级为纯文本显示
    def _inject_base_styles(self):
        base_style = """
        <style>
            .paragraph-fade { animation: paragraph-fade 0.3s; }
            .word-fade { animation: word-fade 0.15s; }
        </style>
        """
        cursor = self.text_browser.textCursor()
        cursor.insertHtml(base_style)

    def _should_auto_scroll(self):
        """判断是否需要自动滚动"""
        scrollbar = self.text_browser.verticalScrollBar()
        return scrollbar.value() + scrollbar.pageStep() >= scrollbar.maximum() - 10

    def _process_text(self, text):
        """增强型文本处理器"""
        from html import escape

        # HTML 转义
        processed = escape(text)

        # 检查是否包含中日韩字符
        has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF or 0x3040 <= ord(c) <= 0x30FF for c in text)
        
        if not has_cjk:
            # 对非 CJK 文本中的空格进行换行优化
            processed = processed.replace(' ', '<wbr>')

        # 保留连续空格
        processed = processed.replace(' ', '&nbsp;')

        # 中文标点换行优化
        processed = processed.replace('。', '。<wbr>').replace('，', '<wbr>，')

        # 英文单词断字
        processed = processed.replace('-', '<wbr>-')

        return processed

    def _smooth_scroll_to_bottom(self):
        """动画滚动实现"""
        anim = QPropertyAnimation(self.text_browser.verticalScrollBar(), b"value")
        anim.setDuration(300)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.setStartValue(self.text_browser.verticalScrollBar().value())
        anim.setEndValue(self.text_browser.verticalScrollBar().maximum())
        anim.start()
    def append_text(self, text):
        scrollbar = self.text_browser.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        
        # 插入内容
        self.text_browser.append(text)
        
        # 自动滚动到底部
        if at_bottom:
            self.text_browser.moveCursor(QTextCursor.End)
            self.text_browser.ensureCursorVisible()
    
    def importFile(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            '导入文件', 
            '', 
            '文档 (*.pdf *.doc *.docx);;数据表 (*.csv *.xls *.xlsx);;图片 (*.jpg *.png *.jpeg);;所有文件 (*)'
        )
        
        # 空选择校验
        if not paths:
            QMessageBox.information(self, "提示", "未选择任何文件")
            return

        # 文件类型二次校验
        valid_exts = ('.pdf', '.doc', '.docx', '.jpg', '.png', '.jpeg')
        valid_paths = [p for p in paths if p.lower().endswith(valid_exts)]
        
        if not valid_paths:
            QMessageBox.warning(self, "错误", 
                "未选择支持的文档格式（支持：PDF/Word/图片）")
            return

        try:
            # 创建进度对话框
            progress_dialog = QProgressDialog(
                "加载文件中...", 
                "取消", 
                0, 
                len(valid_paths) * 4,  # 每个文件4个步骤
                self
            )
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setWindowTitle("文件加载进度")
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.setMinimumDuration(0)

            # 设置进度回调
            self.doc_manager.set_progress_callback(
                lambda step: self._update_progress(progress_dialog, step)
            )

            # 显示加载动画
            self.loading_gif.start()
            self.loading_label.show()

            # 设置模式并添加文档
            self.doc_manager.set_mode(len(valid_paths))
            
            for index, path in enumerate(valid_paths):
                if progress_dialog.wasCanceled():
                    break

                # 更新全局进度信息
                progress_dialog.setLabelText(
                    f"正在处理文件 ({index+1}/{len(valid_paths)})\n"
                    f"{os.path.basename(path)}"
                )

                retry_count = 0
                while retry_count < 3:
                    try:
                        self.doc_manager.add_documents([path])
                        self._update_file_list_display(path)
                        break
                    except chromadb.errors.IDAlreadyExistsError:
                        print(f"文档已存在，跳过: {path}")
                        break
                    except Exception as e:
                        if "index" in str(e) and "str" in str(e):
                            print("检测到元数据损坏，尝试重建集合...")
                            self.doc_manager.rebuild_collection()
                            retry_count += 1
                        else:
                            raise
                    QApplication.processEvents()

            # 完成特效
            if not progress_dialog.wasCanceled():
                self._show_complete_effect()

        except Exception as e:
            QMessageBox.critical(self, "系统错误", 
                f"文件加载过程发生严重错误:\n{str(e)}")
        finally:
            # 清理资源
            self.doc_manager.set_progress_callback(None)
            progress_dialog.close()
            self.loading_gif.stop()
            self.loading_label.hide()
            self.doc_count_label.setText(f"已加载文档: {len(self.doc_manager.loaded_paths)}")

        if valid_paths and not progress_dialog.wasCanceled():
            first_path = valid_paths[0]
            self._show_document_info(first_path)
            self.file_list_widget.setCurrentRow(0)

    def _update_progress(self, dialog, step_info):
        """统一处理进度更新"""
        if step_info.get('type') == 'start':
            # 初始化文件进度
            self.current_file_steps = step_info['total_steps']
            self.current_step = 0
        else:
            # 更新步骤进度
            self.current_step += 1
            total = dialog.maximum()
            current = dialog.value() + 1
            
            # 计算百分比
            percent = min(int((current / total) * 100), 100)
            
            # 更新对话框
            dialog.setValue(current)
            dialog.setLabelText(
                f"{dialog.labelText()}\n"
                f"当前步骤: {step_info['message']} ({percent}%)"
            )
        QApplication.processEvents()

    def _show_complete_effect(self):
        """加载完成动画效果"""
        effect = QGraphicsOpacityEffect(self.file_list_widget)
        self.file_list_widget.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(1200)
        anim.setStartValue(0.2)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutBack)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

        # 添加完成提示音
        QApplication.beep()

    def _handle_image_file(self, path):
        print(f"检测到图片文件: {path}")
        try:
            self.file_viewer.load(QUrl.fromLocalFile(path))
            QMessageBox.information(self, "图片加载", "图片已成功加载预览")
        except Exception as e:
            QMessageBox.warning(self, "图片错误", f"无法加载图片: {str(e)}")

    def _update_file_list_display(self, path):
        """在侧边栏显示已加载文件"""
        # 确保列表部件已初始化
        if not hasattr(self, 'file_list_widget') or self.file_list_widget is None:
            self.file_list_widget = QListWidget()
            self.file_list_widget.itemClicked.connect(self.on_file_selected)
            self.splitter.insertWidget(0, self.file_list_widget)
        
        # 获取已加载的文件路径列表
        loaded_paths = self.doc_manager.loaded_paths
        # 清空现有列表
        self.file_list_widget.clear()
        # 避免重复添加
        for path in self.doc_manager.loaded_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.file_list_widget.addItem(item)

        if self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            self.on_file_selected(self.file_list_widget.item(0))  # 触发预览更新


    def _show_error(self, message):
        """显示错误信息"""
        self.file_viewer.setHtml(f"<p style='color: red; padding: 20px'>{message}</p>")

    def _preview_file(self, path):
        """统一处理文件预览"""
        print(f'Previewing file: {path}')
        self.path = path
        
        try:
            if is_image(path):
                self.file_viewer.load(QUrl.fromLocalFile(path))
            elif path.lower().endswith('.pdf'):
                self.file_viewer.load(QUrl.fromLocalFile(path))
            elif path.lower().endswith(('.csv', '.xls', '.xlsx')):
                try:
                    self.df = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
                    html = self.df.to_html(classes='table table-striped', border=0)
                    self.file_viewer.setHtml(f"<style>table {{width: 100%}}</style>{html}")
                except Exception as e:
                    self._show_error(f"数据文件读取失败: {str(e)}")
            
            elif path.lower().endswith(('.doc', '.docx')):
                # 自定义CSS样式
                custom_css = """
                <style>
                    body {
                        font-family: 'Microsoft YaHei', Arial, sans-serif;
                        line-height: 1.6;
                        margin: 2em;
                        color: #333;
                        background: #fff;
                    }
                    article.doc-preview {
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1 { 
                        font-size: 1.8em; 
                        border-bottom: 2px solid #eee; 
                        padding-bottom: 0.3em; 
                        margin: 1em 0 0.5em;
                    }
                    h2 { font-size: 1.6em; color: #444; margin: 1.2em 0 0.6em; }
                    h3 { font-size: 1.4em; color: #555; margin: 1em 0 0.5em; }
                    p { margin: 0.8em 0; }
                    ul, ol { 
                        margin: 0.8em 0;
                        padding-left: 2em;
                    }
                    table.doc-table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 1.5em 0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        background: white;
                    }
                    th, td {
                        border: 1px solid #ddd;
                        padding: 10px;
                        text-align: left;
                        vertical-align: top;
                    }
                    th {
                        background-color: #f8f9fa;
                        font-weight: 600;
                    }
                    img {
                        max-width: 100% !important;
                        height: auto !important;
                        margin: 1em 0;
                        border-radius: 4px;
                    }
                    .code { 
                        background: #f5f5f5; 
                        padding: 1em;
                        border-radius: 4px;
                        font-family: Consolas, monospace;
                    }
                    .footnote {
                        font-size: 0.9em;
                        color: #666;
                        border-top: 1px solid #eee;
                        margin-top: 2em;
                        padding-top: 1em;
                    }
                </style>
                """

                # 增强样式映射
                style_map = """
                p[style-name='Title'] => h1.doc-title:fresh
                p[style-name='Heading 1'] => h2.section-title:fresh
                p[style-name='Heading 2'] => h3.subsection-title:fresh
                p[style-name='Heading 3'] => h4.child-title:fresh
                p[style-name='Subtitle'] => h2.subtitle:fresh
                p[style-name='toc 1'] => h3.toc-level1:fresh
                p[style-name='toc 2'] => h4.toc-level2:fresh
                p[style-name='toc 3'] => h5.toc-level3:fresh
                p[style-name='caption'] => p.caption:fresh
                p[style-name='List Paragraph'] => li.list-item
                p[style-name='footnote text'] => div.footnote
                table => table.doc-table:rename(style)
                r => span.text-run
                """

                def image_handler(image):
                    """处理文档内嵌图片"""
                    try:
                        with image.open() as image_bytes:
                            encoded = base64.b64encode(image_bytes.read()).decode('utf-8')
                            return {
                                "src": f"data:{image.content_type};base64,{encoded}",
                                "alt": image.alt_text or "Document Image",
                                "style": "max-width: 100%; height: auto; margin: 10px 0;",
                                "class": "doc-image"
                            }
                    except Exception as e:
                        print(f"图片处理失败: {str(e)}")
                        return {"src": "about:blank"}

                try:
                    with open(path, "rb") as doc_file:
                        # 执行转换
                        result = mammoth.convert_to_html(
                            doc_file,
                            style_map=style_map,
                            convert_image=mammoth.images.img_element(image_handler),
                            ignore_empty_paragraphs=False,
                            include_default_style_map=True
                        )

                        # 后处理HTML
                        html_content = post_process_html(result.value)
                        messages = "<br>".join(str(m) for m in result.messages) if result.messages else ""

                        # 构建完整HTML
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            {custom_css}
                        </head>
                        <body>
                            <article class="doc-preview">
                                {html_content}
                            </article>
                            <div style="color: #999; margin: 2em 0; padding-top: 1em; border-top: 1px solid #eee">
                                转换消息: {messages}
                            </div>
                        </body>
                        </html>
                        """

                        # 调试输出
                        debug_path = os.path.join(tempfile.gettempdir(), "doc_preview_debug.html")
                        with open(debug_path, "w", encoding="utf-8") as f:
                            f.write(full_html)
                        print(f"调试文件已保存至: {debug_path}")

                        self.file_viewer.setHtml(full_html)

                except mammoth.DocxFileNotFoundError:
                    self._show_error("文档文件不存在或路径错误")
                except mammoth.DocxInvalidXml:
                    self._show_error("文档包含非法XML结构")
                except mammoth.UnsupportedImageFormat as e:
                    self._show_error(f"不支持的图片格式: {str(e)}")
                except Exception as e:
                    self._show_error(f"未知错误: {str(e)}")

            else:
                self.file_viewer.setHtml(f"<p>不支持预览格式: {os.path.basename(path)}</p>")

        except Exception as e:
            print(f'预览失败: {str(e)}')
            self.file_viewer.setHtml(f"<p style='color:red; padding: 20px'>预览错误: {str(e)}</p>")

    def on_file_selected(self, item):
        """点击文件列表时触发"""
        try:
            if not item:
                return
            selected_path = item.data(Qt.UserRole)
            if not os.path.exists(selected_path):
                QMessageBox.warning(self, "路径错误", "文件不存在或已被移动")
                return

            self.path = selected_path
            self.current_file = selected_path  # 显式设置当前文件
            # 更新文档信息显示
            self._show_document_info(selected_path)
            self._preview_file(selected_path)
            self.doc_manager.set_current_document(selected_path)
            print(f"已切换至文档向量库：{selected_path}")
            self._load_qa_history(selected_path)
            # 异步加载文档向量库
            QTimer.singleShot(0, lambda: 
                self._async_load_document(selected_path)
            )
            
            #更新文档计数
            self.doc_count_label.setText(f"已加载文档: {len(self.doc_manager.loaded_paths)}")
        except Exception as e:
            QMessageBox.warning(self, "文档切换错误", 
                f"无法加载文档向量库：\n{str(e)}")
            return
        
        # 显示文档基本信息
        self._show_document_info(selected_path)
        
        # 加载该文档的问答历史
        self._load_qa_history(selected_path)
        # 初始化当前文件的对话历史
        if self.current_file not in self.messages:
            self.messages[self.current_file] = []

    def _async_load_document(self, path):
        """异步加载文档向量库"""
        try:
            self.doc_manager.set_current_document(path)
            print(f"已切换至文档向量库：{path}")
            self._load_qa_history(path)
            
            # 触发界面更新信号
            QMetaObject.invokeMethod(
            self,
            "_show_document_info",
            Qt.QueuedConnection,
            Q_ARG(str, path)  # 传递当前路径
        )
            
        except Exception as e:
            error_msg = f"文档加载失败: {str(e)}"
            QMetaObject.invokeMethod(
                self.preview_info_label,
                "setText",
                Qt.QueuedConnection,
                Q_ARG(str, error_msg)
            )

    def sendMessage(self):
        # 用户输入样式
        input_html = f'''
        <div style="
            color: #6A8759;
            border-left: 3px solid #6A8759;
            margin: 8px 0;
            padding: 6px;
        ">
            ➜ {self.input_text_edit.toPlainText()}
        </div>
        '''
        self.text_browser.append(input_html)

        print('send message')
        self.new_reply = ''

        if hasattr(self, 'current_chat_thread') and self.current_chat_thread is not None:
            if self.current_chat_thread.isRunning():
                self.current_chat_thread.stop()
                self.current_chat_thread.wait(2000)
    
        # 新增状态重置
        self.current_response = ""
        self.text_buffer = ""

        if not hasattr(self, 'messages') or not isinstance(self.messages, dict):
            self.messages = {}

        # 获取输入框的文本
        current_file_path = self.get_current_file_path()
        if not current_file_path:
            QMessageBox.warning(self, "警告", "请先选择一个文件。")
            return
        
        # 确保消息列表初始化
        if current_file_path not in self.messages:  
            self.messages[current_file_path] = []
        
        # 调用Ollama的接口，获取回复文本
        self.input_text = self.input_text_edit.toPlainText().strip()
        if not self.input_text:
            QMessageBox.warning(self, "输入错误", "请输入问题内容")
            return
        
        self.model = self.model_selector.currentText()
        self.role = self.role_selector.currentText()

        # 强制添加用户消息
        self.messages[current_file_path].append({
            'role': self.role,
            'content': self.input_text,
            'tip': '用户最新提问'  # 可根据实际情况调整
        })
        
        # 显示用户消息
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_browser.append(
            f'<div style="color:black; margin:5px 0;">'
            f'[{timestamp}] 💬用户提问：<br/>'
            f'{self.input_text}'
            f'</div>'
        )

        if self.memory_selector.currentText() == 'Input':
            if self.messages[current_file_path] == []:
                self.messages[current_file_path].append(
                {
                    'role': self.role,
                    'content': self.input_text,
                    'tip':''
                }
                )
            else:
                self.messages[current_file_path].append(
                {
                    'role': self.role,
                    'content': self.input_text,
                    'tip':'This is the history of the conversation, please do not reply to the previous messages. Remember the input but do not repeat the previous messages.'
                }
                )
        else:
            if self.messages[current_file_path] == []:
                self.messages[current_file_path].append(
                {
                    'role': self.role,
                    'content': self.input_text,
                    'tip':''
                }
                )
            else:
                self.messages[current_file_path].append(
                {
                    'role': self.role,
                    'content': self.input_text,
                    'tip':'This is the history of the conversation, please do not reply to the previous messages. But do remember all the conversations.'
                }
                )
        
        # 构建增强提示（兼容两种记忆模式）
        context_messages = self._build_contextual_prompt(current_file_path)  # 传递当前文件路径
        # 修改发送消息时的提示模板
        # 获取对话上下文
        session_id = f"multi_{hash(frozenset(self.active_links))}" if self.doc_manager.current_mode == "collection" else current_file_path
        conversation_history = self.doc_manager.get_conversation_context(session_id)
        
        # 构建增强提示模板
        template = """请结合以下文档内容和对话历史回答问题：
        
        【相关文档内容】
        {doc_context}
        
        【对话历史】
        {conversation_history}
        
        要求：
        1. 保持回答与先前对话的连贯性
        2. 若涉及多个文档需注明来源
        3. 回答需与之前的结论逻辑一致
        
        问题：{question}
        """
        
        full_prompt = template.format(
            doc_context=context_messages['doc_context'],
            conversation_history=conversation_history,
            question=self.input_text
        )

        # 启动聊天线程前先停止现有线程
        # 在启动新线程前终止旧线程
        if self.current_chat_thread and self.current_chat_thread.isRunning():
            self.current_chat_thread.stop()  # 停止方法
            self.current_chat_thread.quit()
            self.current_chat_thread.wait(2000)
    
        # 获取所有已加载文件路径
        all_paths = list(self.doc_manager.single_docs.keys()) 
        
        # 获取当前活动文档路径
        current_path = self.get_current_file_path()
        if not current_path:
            QMessageBox.warning(self, "错误", "请先选择要分析的文档")
            return
    
        # 构建增强提示时明确指定当前文档
        context_messages = self._build_contextual_prompt(current_path)
        # 启动聊天线程
        self.current_chat_thread = ChatThread(
            prompt=full_prompt,
            messages=self.messages[current_file_path],
            document_manager=self.doc_manager,
            model=self.model,
            paths=[current_path],   # 传递所有已加载路径
            context_session=session_id
        )

        # 连接线程结束信号
        self.current_chat_thread.finished_with_response.connect(self.on_chat_finished)  # 连接到新的信号
        self.current_chat_thread.new_text.connect(self.update_text_browser)
        self.current_chat_thread.start()
        
        self.input_text_edit.clear()
        

    def _build_contextual_prompt(self, file_path):
        context = {
            'history': '',
            'doc_context': ''
        }
        
        # 获取对话历史
        if file_path and file_path in self.messages:
            context['history'] = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.messages[file_path]]
            )
        
        # 获取文档上下文
        if file_path and hasattr(self, 'input_text'):
            try:
                context['doc_context'] = self.doc_manager.get_combined_context(
                    self.input_text  # 使用当前输入作为检索问题
                )
            except Exception as e:
                print(f"Error getting document context: {e}")
    
        return context

    def newChat(self):
        # 新建对话
        # self.output_text_list = []
        self.show_text = ''
        self.path = ''  # 加载文件路径
        self.messages = []
        self.output_text_edit.clear()
        self.input_text_edit.clear()
        self.file_viewer.setUrl(QUrl())
        self.text_browser.clear()
        print('new chat')
        
    def openChat(self):
        # 弹出文件对话框，获取保存文件的路径
        path, _ = QFileDialog.getOpenFileName(self, 'Open Chat', 'clap', 'Clap Save Files (*.clap);;All Files (*)')
        
        # 检查文件路径是否为空
        if path != '':
            # 以读取模式打开文件
            with open(path, 'rb') as f:
                # 使用pickle模块加载字典
                data = pickle.load(f)
                # 将字典的内容赋值给相应的变量
                # self.output_text_list = data['output_text_list']
                self.show_text = data['show_text']
                self.messages = data['messages']
                self.model_selector.setCurrentText(data['model'])
                self.role_selector.setCurrentText(data['role'])
                try:
                    self.path = data['path']
                except:
                    self.path = ''
                
                self.text_browser.setText(self.show_text) # 将文本添加到文本浏览器中
    
        # self.sendMessage()
    
    def saveChat(self):
        # 弹出文件对话框，获取保存文件的路径
        path, _ = QFileDialog.getSaveFileName(self, 'Save Chat', 'clap', 'Clap Save Files (*.clap);;All Files (*)')
        
        # 检查文件路径是否为空
        if path != '':
            # 以写入模式打开文件
            with open(path, 'wb') as f:
                # 创建一个字典，包含要保存的变量
                data = {
                    # 'output_text_list': self.output_text_list,
                    'show_text': self.show_text,
                    'messages': self.messages,
                    'model' : self.model_selector.currentText(),
                    'role' : self.role_selector.currentText(),
                    'path': self.path
                }
                # 使用pickle模块保存字典
                pickle.dump(data, f)

    def exportMarkdown(self):
        
        # 弹出文件对话框，获取保存文件的路径
        path, _ = QFileDialog.getSaveFileName(self, 'To Markdown', 'clap', 'Text Files (*.md);;All Files (*)')
        
        # 获取输出框的文本
        if (path != ''):
            with open(path, 'w', encoding='utf-8') as f:
                text_to_write = self.show_text
                f.write(text_to_write + '\n')
            print('save chat')
    
    def display_qa(self, qa_list):
        self.output_text_edit.clear()
        for entry in qa_list:
            role = entry['role'].capitalize()
            content = entry['content']
            self.output_text_edit.append(f"<strong>{role}:</strong> {content}")

    def get_current_file_path(self):
        try:
            if not hasattr(self, 'file_list_widget'):
                return None
            
            selected_items = self.file_list_widget.selectedItems()
            
            if not selected_items:
                # 尝试获取最后添加的文件
                if self.file_list_widget.count() > 0:
                    last_item = self.file_list_widget.item(self.file_list_widget.count()-1)
                    return last_item.data(Qt.UserRole)
                return None
            
            return selected_items[0].data(Qt.UserRole)
        
        except Exception as e:
            print(f"获取当前文件路径出错: {str(e)}")
            return None

    def _load_qa_for_pdf(self, file_path):
        qa_list = self.doc_manager.get_questions_answers(file_path)
        self.display_qa(qa_list)

def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QApplication.setApplicationName(metadata["Formal-Name"])

    app = QApplication(sys.argv)
    main_window = ChatLocalAndPersistent()
    sys.exit(app.exec())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ChatLocalAndPersistent()
    main_window.show()  # 显示主窗口
    sys.exit(app.exec())