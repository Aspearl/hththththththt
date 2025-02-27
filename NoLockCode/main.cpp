#include <atomic>
#include <memory>
#include <iostream>
#include <thread>
#include <vector>
#include <WinSock2.h>
#include <Windows.h>
#include <string>

#pragma comment(lib, "ws2_32.lib")
constexpr auto PORT = 12345;
constexpr auto BUFFER_SIZE = 1024;

template <typename T>
class LockFreeQueue {
public:
    LockFreeQueue(size_t size) : size_(size), head_(0), tail_(0) {
        data_ = new T[size_];
    }

    ~LockFreeQueue() {
        delete[] data_;
    }

    bool enqueue(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) % size_;
        if (next_tail != head_.load(std::memory_order_acquire)) {
            data_[tail] = item;
            tail_.store(next_tail, std::memory_order_release);
            return true;
        }
        return false; // Queue full
    }

    bool dequeue(T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head != tail_.load(std::memory_order_acquire)) {
            item = data_[head];
            head_.store((head + 1) % size_, std::memory_order_release);
            return true;
        }
        return false; // Queue empty
    }

private:
    T* data_;
    const size_t size_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};


// A simple function to handle incoming data (placeholder for actual processing)
void process_data(const std::string& data) {
    std::cout << "Processing: " << data << std::endl;
}

// TCP Server thread function
void server_thread(LockFreeQueue<std::string>& queue) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = { 0 };

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen)) < 0) {
            perror("Accept failed");
            continue;
        }

        int valread = recv(new_socket, buffer, BUFFER_SIZE, 0);
        if (valread > 0) {
            std::string data(buffer, valread);
            if (queue.enqueue(data)) {
                std::cout << "Data enqueued: " << data << std::endl;
            }
            else {
                std::cout << "Queue full, dropping data: " << data << std::endl;
            }
        }
        closesocket(new_socket);
    }
}

// Data processing thread
void processing_thread(LockFreeQueue<std::string>& queue) {
    while (true) {
        std::string data;
        if (queue.dequeue(data)) {
            process_data(data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    LockFreeQueue<std::string> queue(10); // Queue size 10

    // Start server thread to handle TCP connections
    std::thread server(server_thread, std::ref(queue));
    std::thread processor(processing_thread, std::ref(queue));

    server.join();
    processor.join();

    return 0;
}


