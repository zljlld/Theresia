import requests
import json
import time

def call_deepseek_api(
    api_key: str,
    question: str,
    model: str = "deepseek-chat",
    stream: bool = True,  # 是否流式输出（逐字返回）
    chat_history: list = None,  # 对话历史，用于多轮对话
    max_history_rounds: int = 15,  # 最大保留的对话轮次，每轮包含user和assistant两条消息
    max_history_age: int = 43200,  # 最大保留的对话时间，单位：秒（默认12小时）
    system_prompt: str = None  # 系统提示词，用于设置AI的角色和行为
) -> tuple[str, list]:
    """
    调用DeepSeek API，开启推理模式，支持流式输出和多轮对话
    
    Args:
        api_key: 你的DeepSeek API Key（从平台获取）
        question: 要提问的问题
        model: 使用的模型名称，如deepseek-chat
        stream: 是否开启流式输出（True=逐字返回，更流畅）
        chat_history: 对话历史列表，包含之前的对话内容
        max_history_rounds: 最大保留的对话轮次，超过则自动截断
        max_history_age: 最大保留的对话时间，超过则自动截断（单位：秒）
        system_prompt: 系统提示词，用于设置AI的角色和行为
    
    Returns:
        tuple[str, list]: 模型的回答内容和更新后的对话历史
    """
    # API基础配置
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 获取当前时间戳
    current_time = time.time()
    
    # 初始化对话历史
    if chat_history is None:
        chat_history = []
    else:
        # 1. 基于时间的对话历史截断
        # 检查对话历史中最早的消息时间
        if chat_history and "timestamp" in chat_history[0]:
            earliest_time = chat_history[0]["timestamp"]
            # 如果最早的消息超过max_history_age秒，则清空历史
            if current_time - earliest_time > max_history_age:
                chat_history = []
    
    # 如果提供了系统提示词，且对话历史中还没有系统提示词，则添加
    if system_prompt and not any(msg["role"] == "system" for msg in chat_history):
        # 将<time>替换为真实时间，格式：YYYY-MM-DD HH:MM:SS
        real_time = time.strftime("%Y-%m-%d %H:%M:%S")
        updated_system_prompt = system_prompt.replace("<time>", real_time)
        # 系统提示词添加到对话历史的最前面，并添加时间戳
        chat_history.insert(0, {"role": "system", "content": updated_system_prompt, "timestamp": current_time})
    
    # 添加当前问题到对话历史，包含时间戳
    chat_history.append({"role": "user", "content": question, "timestamp": current_time})
    
    # 2. 基于轮次的对话历史截断：只保留最近的max_history_rounds轮对话
    # 每轮对话包含user和assistant两条消息，所以总数是max_history_rounds * 2
    # 注意：系统提示词不计入轮次限制，所以这里只截断用户和助手的对话
    user_assistant_messages = [msg for msg in chat_history if msg["role"] != "system"]
    if len(user_assistant_messages) > max_history_rounds * 2:
        # 找到需要保留的最早消息索引
        keep_from_index = len(chat_history) - len(user_assistant_messages) + (max_history_rounds * 2)
        chat_history = chat_history[:len(chat_history) - len(user_assistant_messages)] + user_assistant_messages[-(max_history_rounds * 2):]
    
    # 构建发送给API的messages，移除时间戳字段
    # DeepSeek API不接受timestamp字段，所以需要过滤掉
    api_messages = []
    for msg in chat_history:
        # 只保留role和content字段，移除timestamp
        api_msg = {k: v for k, v in msg.items() if k in ["role", "content"]}
        api_messages.append(api_msg)
    
    # 核心参数（重点：开启推理+支持多轮对话）
    payload = {
        "model": model,
        "messages": api_messages,  # 使用过滤后的对话历史，支持多轮对话
        "think": True,           # 核心：开启DeepSeek的推理/思考模式
        "max_new_tokens": 200,   # 增加输出长度，支持更详细的回答
        "temperature": 0.7,      # 适当增加随机性，提高回答质量
        "top_p": 0.9,            # 扩大采样范围，增加回答多样性
        "stream": stream,        # 开启流式输出，逐字返回
        "stop": ["\n\n"]        # 调整停止条件，允许推理和步骤
    }

    try:
        # 发送请求
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误（如401/404）

        # 处理非流式响应
        if not stream:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            # 将模型回答添加到对话历史，包含时间戳
            chat_history.append({"role": "assistant", "content": answer, "timestamp": current_time})
            return answer, chat_history
        
        # 处理流式响应
        else:
            answer = ""
            print("Theresia：", end="", flush=True)  # 提示开始回答
            for chunk in response.iter_lines():
                if chunk:
                    chunk_data = chunk.decode("utf-8").lstrip("data: ")
                    if chunk_data != "[DONE]":
                        try:
                            chunk_json = json.loads(chunk_data)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            answer += delta
                            print(delta, end="", flush=True)  # 实时打印
                        except json.JSONDecodeError:
                            continue
            print()  # 换行
            # 将模型回答添加到对话历史，包含时间戳
            chat_history.append({"role": "assistant", "content": answer, "timestamp": current_time})
            return answer, chat_history

    # 异常处理（覆盖常见错误）
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP错误：{e}，响应内容：{response.text}"
        return error_msg, chat_history
    except requests.exceptions.Timeout:
        error_msg = "请求超时：API响应时间超过30秒，请检查网络或缩短max_new_tokens"
        return error_msg, chat_history
    except requests.exceptions.ConnectionError:
        error_msg = "连接错误：无法连接到DeepSeek API，请检查网络"
        return error_msg, chat_history
    except Exception as e:
        error_msg = f"未知错误：{str(e)}"
        return error_msg, chat_history

# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 替换为你的API Key（从https://platform.deepseek.com/获取）
    YOUR_API_KEY = "sk-ed73e601ff574217839a9568bc902498"
    
    # 初始读取提示词
    current_prompt = ""
    last_prompt_content = ""  # 用于保存上一次读取的提示词内容
    
    def check_and_remove_expired_memory():
        """检查并删除过期的记忆"""
        try:
            # 读取完整的提示词文件内容
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                full_content = f.read()
            
            # 找到初始提示词结束标记
            end_prompt_marker = "===END_INITIAL_PROMPT==="
            end_marker_pos = full_content.find(end_prompt_marker)
            if end_marker_pos == -1:
                return  # 如果没有标记，不处理
            
            # 获取初始提示词部分
            initial_prompt = full_content[:end_marker_pos]
            rest_content = full_content[end_marker_pos:]
            
            # 分割初始提示词为行
            lines = initial_prompt.split("\n")
            updated_lines = []
            
            # 获取当前时间戳
            current_timestamp = time.time()
            
            for line in lines:
                # 检查是否是记忆行，格式：<时间> 内容 结尾带有<年/月/日/小时:分钟>
                if line.strip().startswith("<"):
                    # 查找记忆截至时间（行尾的<年/月/日/小时:分钟>）
                    if line.strip().endswith(">"):
                        end_time_pos = line.rfind("<")
                        if end_time_pos != -1:
                            end_time_str = line[end_time_pos+1:-1]
                            try:
                                # 解析时间格式：年/月/日/小时:分钟 -> %Y/%m/%d/%H:%M
                                expire_time = time.strptime(end_time_str, "%Y/%m/%d/%H:%M")
                                expire_timestamp = time.mktime(expire_time)
                                # 如果记忆未过期，保留
                                if expire_timestamp > current_timestamp:
                                    updated_lines.append(line)
                            except ValueError:
                                # 如果时间格式不正确，保留该行
                                updated_lines.append(line)
                    else:
                        # 如果行尾没有正确的时间格式，保留该行
                        updated_lines.append(line)
                else:
                    # 非记忆行，直接保留
                    updated_lines.append(line)
            
            # 重新构建初始提示词
            updated_initial_prompt = "\n".join(updated_lines)
            
            # 重新构建完整内容
            updated_content = updated_initial_prompt + rest_content
            
            # 写入更新后的内容
            with open("system_prompt.txt", "w", encoding="utf-8") as f:
                f.write(updated_content)
                
        except Exception as e:
            pass  # 静默处理错误，不影响主程序运行
    
    def read_prompt_from_file():
        """从文件中读取提示词，只保留分隔符之前的内容"""
        try:
            # 先检查并删除过期记忆
            check_and_remove_expired_memory()
            
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                content = f.read()
                # 查找分隔符，只保留分隔符之前的内容
                if "===END_INITIAL_PROMPT===" in content:
                    return content.split("===END_INITIAL_PROMPT===")[0].strip()
                # 如果没有分隔符，返回全部内容
                return content.strip()
        except FileNotFoundError:
            print("警告：未找到system_prompt.txt文件，将使用默认提示词")
            return "你是一个AI助手，使用中文交流。"
        except Exception as e:
            print(f"读取提示词文件时发生错误：{e}，将使用默认提示词")
            return "你是一个AI助手，使用中文交流。"
    
    # 首次读取提示词
    current_prompt = read_prompt_from_file()
    last_prompt_content = current_prompt
    print("已成功读取初始提示词")
    
    print("===== DeepSeek API 对话交互 =====")
    print("提示：输入 'exit' 或 '退出' 结束对话")
    
    # 初始化对话历史，用于保存上下文
    chat_history = None
    
    while True:
        # 获取用户输入的问题
        question = input("\n杜：").strip()
        
        # 检查是否退出
        if question.lower() in ["exit", "退出"]:
            print("\n===== 对话结束 =====")
            break
        
        # 检查是否为空输入
        if not question:
            print("问题不能为空，请重新输入")
            continue
        
        # 读取最新的提示词文件
        latest_prompt = read_prompt_from_file()
        
        # 检查提示词是否发生改变
        if latest_prompt != last_prompt_content:
            print("\n提示词已更新，将重置对话历史并使用新的提示词")
            current_prompt = latest_prompt
            last_prompt_content = latest_prompt
            # 重置对话历史，使用新的提示词
            chat_history = None
        
        # ===== 记忆处理逻辑 =====
        def process_memory():
            """处理记忆提示词逻辑"""
            try:
                # 1. 读取完整的提示词文件内容
                with open("system_prompt.txt", "r", encoding="utf-8") as f:
                    full_content = f.read()
                
                # 2. 找到记忆提示词的位置
                memory_prompt_start = full_content.find("# 记忆提示词")
                if memory_prompt_start == -1:
                    return
                
                # 3. 找到记忆提示词的下一行
                memory_prompt_line = full_content.find("\n", memory_prompt_start)
                if memory_prompt_line == -1:
                    memory_prompt_text = ""
                else:
                    next_line_start = memory_prompt_line + 1
                    next_line_end = full_content.find("\n", next_line_start)
                    if next_line_end == -1:
                        memory_prompt_text = full_content[next_line_start:].strip()
                    else:
                        memory_prompt_text = full_content[next_line_start:next_line_end].strip()
                
                # 4. 读取之前对话的记忆内容（"以下是之前对话的记忆"以后"===END_INITIAL_PROMPT==="以前）
                previous_memories = ""
                memories_start = full_content.find("以下是之前对话的记忆")
                end_marker = "===END_INITIAL_PROMPT==="
                memories_end = full_content.find(end_marker)
                
                if memories_start != -1 and memories_end != -1 and memories_start < memories_end:
                    # 提取记忆内容
                    memories_content = full_content[memories_start:memories_end].strip()
                    # 只保留记忆行，过滤掉标题行
                    memories_lines = memories_content.split("\n")[1:]  # 去掉第一行"以下是之前对话的记忆"
                    previous_memories = "\n".join([line.strip() for line in memories_lines if line.strip()])
                
                # 5. 准备最后5轮对话作为上下文
                context_text = ""
                if chat_history:
                    # 过滤出用户和助手的对话（排除系统提示词）
                    user_assistant_msgs = [msg for msg in chat_history if msg["role"] != "system"]
                    # 每轮包含user和assistant两条消息，获取最后5轮（10条消息）
                    last_5_rounds = user_assistant_msgs[-10:] if len(user_assistant_msgs) > 10 else user_assistant_msgs
                    
                    # 构建上下文文本
                    for msg in last_5_rounds:
                        role = "用户" if msg["role"] == "user" else "助手"
                        context_text += f"{role}：{msg['content']}\n"
                
                # 6. 将之前的记忆、上下文、用户输入拼接在记忆提示词后
                full_question = f"{memory_prompt_text}"
                if previous_memories:
                    full_question += f"\n\n之前的记忆：\n{previous_memories}"
                if context_text:
                    full_question += f"\n\n上下文：\n{context_text}"
                full_question += f"\n\n用户当前输入：{question}"
                
                # 5. 单独请求API，不使用上下文
                memory_answer, _ = call_deepseek_api(
                    YOUR_API_KEY, 
                    full_question, 
                    stream=False, 
                    chat_history=None,  # 不使用上下文
                    system_prompt=""
                )
                
                # 6. 处理API返回结果
                if memory_answer.strip().upper() == "NO":
                    return  # 如果回答是NO，不用理会
                
                # 7. 如果有内容，写入初始提示词中
                if memory_answer.strip():
                    # 获取当前时间，格式：年/月/日/h:m
                    current_time = time.strftime("%Y/%m/%d/%H:%M")
                    # 构建要写入的内容
                    new_memory_line = f"<{current_time}> {memory_answer.strip()}"
                    
                    # 8. 找到===END_INITIAL_PROMPT===的位置，在其之前插入新内容
                    end_prompt_marker = "===END_INITIAL_PROMPT==="
                    end_marker_pos = full_content.find(end_prompt_marker)
                    if end_marker_pos != -1:
                        # 在===END_INITIAL_PROMPT===之前插入新内容和换行
                        updated_content = full_content[:end_marker_pos] + new_memory_line + "\n" + full_content[end_marker_pos:]
                        # 写入更新后的内容
                        with open("system_prompt.txt", "w", encoding="utf-8") as f:
                            f.write(updated_content)
            except Exception as e:
                pass  # 静默处理错误，不影响主程序运行
        
        # 调用API并保存对话历史，传递当前提示词
        start_time = time.time()
        answer, chat_history = call_deepseek_api(
            YOUR_API_KEY, 
            question, 
            stream=True, 
            chat_history=chat_history,
            system_prompt=current_prompt
        )
        end_time = time.time()
        print(f"耗时：{end_time - start_time:.2f}秒")
        
        # 调用记忆处理函数（在主对话之后进行）
        process_memory()
        
        # 可选：打印对话历史长度，用于调试
        # print(f"当前对话历史长度：{len(chat_history)}")