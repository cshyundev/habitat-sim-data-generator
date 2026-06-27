import os
import yaml
from mcap.reader import make_reader

def main():
    print("==================================================")
    print("1. config_stream.yaml 설정 불러오기...")
    with open("config_stream.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    mcap_path = os.path.join(config["output_dir"], config["output_filename"])
    if not os.path.exists(mcap_path):
        print(f"[Error] MCAP 파일이 존재하지 않습니다: {mcap_path}")
        return
        
    print(f"2. MCAP 파일 분석 시작 ({mcap_path})...")
    
    # Track statistics
    topic_counts = {}
    topic_schema_name = {}
    first_log_time = None
    last_log_time = None
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        
        # Iterate through messages
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic not in topic_counts:
                topic_counts[topic] = 0
            topic_counts[topic] += 1
            
            if schema:
                topic_schema_name[topic] = schema.name
            else:
                topic_schema_name[topic] = "Unknown"
            
            log_time = message.log_time
            if first_log_time is None:
                first_log_time = log_time
            last_log_time = log_time
            
    print("--------------------------------------------------")
    print("MCAP 파일 구조 요약:")
    print(f"  - 파일 크기: {os.path.getsize(mcap_path) / (1024 * 1024):.2f} MB")
    
    if first_log_time is not None and last_log_time is not None:
        duration_sec = (last_log_time - first_log_time) / 1e9
        print(f"  - 데이터 시간 범위: {first_log_time / 1e9:.2f} s ~ {last_log_time / 1e9:.2f} s (총 {duration_sec:.2f} 초)")
        
    print("  - 등록된 토픽 및 메시지 개수:")
    for topic, count in topic_counts.items():
        schema_name = topic_schema_name.get(topic, "Unknown")
        print(f"    * Topic: {topic:<10} | Type: {schema_name:<30} | 메시지 수: {count:>4} 개")
        
    print("==================================================")

if __name__ == "__main__":
    main()
