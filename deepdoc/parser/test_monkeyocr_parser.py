#!/usr/bin/env python3
"""
MonkeyOCR底层解析器测试脚本
"""

import os
import sys
import logging
from pathlib import Path

# 强制移除所有旧的 handler，确保日志配置生效
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # 让日志输出到标准输出
)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from deepdoc.parser.monkeyocr_parser import MonkeyOCRParser
except ImportError as e:
    logging.error(f"导入错误: {e}")
    logging.error("请确保在RAGFlow项目根目录下运行此脚本")
    sys.exit(1)


def test_callback(progress, message):
    """测试回调函数"""
    if isinstance(progress, (int, float)) and progress >= 0:
        print(f"[{progress:.1%}] {message}")
    else:
        print(f"[ERROR] {message}")


def test_monkeyocr_parser():
    """测试MonkeyOCR底层解析器"""
    
    logging.info("MonkeyOCR底层解析器测试开始")
    logging.info("=" * 50)
    
    # 测试配置 - 从环境变量或默认值获取
    monkeyocr_url = os.environ.get('MONKEYOCR_URL', 'http://localhost:6006')
    timeout = int(os.environ.get('MONKEYOCR_TIMEOUT', '300'))
    
    # 创建解析器实例
    parser = MonkeyOCRParser(monkeyocr_url=monkeyocr_url, timeout=timeout)
    
    # 测试文件路径
    test_files = [
        '03第3周数学周末练习.pdf',
    ]
    
    # 查找测试文件
    test_file = None
    for filename in test_files:
        if os.path.exists(filename):
            test_file = Path(filename)
            break
    
    if not test_file:
        logging.error("未找到测试文件，请将PDF文件放在当前目录下")
        logging.error("支持的文件名：%s", ', '.join(test_files))
        return
    
    try:
        logging.info(f"测试文件: {test_file}")
        logging.info(f"MonkeyOCR服务: {monkeyocr_url}")
        logging.info(f"超时时间: {timeout}秒")
        logging.info("-" * 50)
        
        # 读取文件
        with open(test_file, 'rb') as f:
            binary_data = f.read()
        
        logging.info(f"文件大小: {len(binary_data):,} 字节")
        
        # 调用底层解析器
        sections, tables = parser(
            filename=str(test_file),
            binary=binary_data,
            from_page=0,
            to_page=100000,
            callback=test_callback
        )
        
        logging.info("=" * 50)
        logging.info("✅ 解析成功!")
        logging.info(f"解析结果: {len(sections)} 个文档片段, {len(tables)} 个表格")
        
        # 显示部分结果
        if sections:
            logging.info("\n📄 文档片段预览:")
            for i, (text, image) in enumerate(sections):
                logging.info(f"片段 {i+1}:")
                logging.info(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                logging.info(f"  图片: {'有' if image else '无'}")
                if image:
                    logging.info(f"  图片尺寸: {image.width}x{image.height}")
        
        if tables:
            logging.info(f"\n📊 表格数量: {len(tables)}")
        
        # 输出统计信息
        total_text_length = sum(len(text) for text, _ in sections)
        images_count = sum(1 for _, image in sections if image)
        
        logging.info(f"\n📈 统计信息:")
        logging.info(f"  总文本长度: {total_text_length:,} 字符")
        logging.info(f"  包含图片的片段: {images_count}")
        logging.info(f"  平均片段长度: {total_text_length // len(sections) if sections else 0} 字符")
        
    except Exception as e:
        logging.error(f"❌ 解析失败: {e}")
        logging.error(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_monkeyocr_integration():
    """测试与应用层的集成"""
    
    logging.info("\n" + "=" * 50)
    logging.info("测试与应用层的集成")
    logging.info("=" * 50)
    
    try:
        from rag.app import naive
        
        # 测试配置 - 从环境变量或默认值获取
        config = {
            'layout_recognize': 'MonkeyOCR',
            'monkeyocr_url': os.environ.get('MONKEYOCR_URL', 'http://localhost:6006'),
            'timeout': int(os.environ.get('MONKEYOCR_TIMEOUT', '300')),
            'chunk_token_num': 512
        }
        
        # 查找测试文件
        test_file = None
        for filename in ['03第3周数学周末练习.pdf']:
            if os.path.exists(filename):
                test_file = Path(filename)
                break
        
        if not test_file:
            logging.warning("未找到测试文件，跳过集成测试")
            return
        
        # 读取文件
        with open(test_file, 'rb') as f:
            binary_data = f.read()
        
        logging.info(f"通过naive应用层测试MonkeyOCR...")
        
        # 通过应用层调用
        result = naive.chunk(
            filename=str(test_file),
            binary=binary_data,
            callback=test_callback,
            parser_config=config,
            lang="Chinese"
        )
        
        logging.info(f"✅ 应用层集成成功!")
        logging.info(f"生成的chunks: {len(result)}")
        
        # 显示chunk示例
        if result:
            logging.info("\n📄 Chunk示例:")
            chunk = result[0]
            logging.info(f"  内容: {chunk.get('content_with_weight', '')[:100]}...")
            logging.info(f"  文档名: {chunk.get('docnm_kwd', '')}")
            logging.info(f"  包含图片: {'是' if chunk.get('image') else '否'}")
        
    except Exception as e:
        logging.error(f"❌ 应用层集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试底层解析器
    success = test_monkeyocr_parser()
    
    # 如果底层解析器测试成功，则测试应用层集成
    # if success:
    #     test_monkeyocr_integration()
    
    logging.info("\n" + "=" * 50)
    logging.info("测试完成")
    logging.info("=" * 50) 