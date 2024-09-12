import time

class AccumulatedTimer:
    def __init__(self):
        self.total_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.total_time += (end_time - self.start_time)

    def reset(self):
        self.total_time = 0.0

    def get_total_time(self):
        return self.total_time

# 使用示例
if __name__ == "__main__":
    timer = AccumulatedTimer()

    with timer:
        time.sleep(1)  # 模拟一些耗时操作

    with timer:
        time.sleep(2)  # 模拟另一些耗时操作

    print(f"Total accumulated time: {timer.get_total_time()} seconds")
