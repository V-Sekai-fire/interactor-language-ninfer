// Erlang C node for a resident NInfer engine. Protocol: docs/cnode-protocol.md in the console repo.
#include <arpa/inet.h>
#include <ei.h>
#include <ei_connect.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ninfer/engine.h"

namespace {

constexpr const char* kHost = "127.0.0.1";

std::mutex send_lock;
std::mutex engine_lock;
std::unique_ptr<ninfer::Engine> engine;
std::atomic<bool> cancel_requested{false};

struct Request {
    int fd;
    erlang_msg message;
    std::string command;
    std::string reference;
    std::vector<std::pair<ninfer::ChatRole, std::string>> messages;
    std::string text;
};

void send_term(const Request& request, ei_x_buff& out) {
    std::lock_guard guard(send_lock);
    ei_send(request.fd, const_cast<erlang_pid*>(&request.message.from), out.buff, out.index);
}

void reply(const Request& request, const char* tag, const std::string* payload) {
    ei_x_buff out;
    ei_x_new_with_version(&out);
    ei_x_encode_tuple_header(&out, 2);
    ei_x_append_buf(&out, request.reference.data(), static_cast<int>(request.reference.size()));
    if (payload == nullptr) {
        ei_x_encode_atom(&out, tag);
    } else {
        ei_x_encode_tuple_header(&out, 2);
        ei_x_encode_atom(&out, tag);
        ei_x_encode_binary(&out, payload->data(), static_cast<long>(payload->size()));
    }
    send_term(request, out);
    ei_x_free(&out);
}

void reply_state(const Request& request, const char* state) {
    ei_x_buff out;
    ei_x_new_with_version(&out);
    ei_x_encode_tuple_header(&out, 2);
    ei_x_append_buf(&out, request.reference.data(), static_cast<int>(request.reference.size()));
    ei_x_encode_tuple_header(&out, 2);
    ei_x_encode_atom(&out, "ok");
    ei_x_encode_atom(&out, state);
    send_term(request, out);
    ei_x_free(&out);
}

void reply_error(const Request& request, const std::string& reason) { reply(request, "error", &reason); }

class DeltaSink final : public ninfer::OutputSink {
public:
    explicit DeltaSink(const Request& request) : request_(request) {}
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel != ninfer::OutputChannel::Content) return;
        ei_x_buff out;
        ei_x_new_with_version(&out);
        ei_x_encode_tuple_header(&out, 3);
        ei_x_encode_atom(&out, "delta");
        ei_x_append_buf(&out, request_.reference.data(), static_cast<int>(request_.reference.size()));
        ei_x_encode_binary(&out, delta.text.data(), static_cast<long>(delta.text.size()));
        send_term(request_, out);
        ei_x_free(&out);
    }

private:
    const Request& request_;
};

void do_load(const Request& request) {
    const char* configured = std::getenv("BONSAI_ARTIFACT");
    const std::string path = request.text.empty() ? (configured ? configured : "") : request.text;
    std::lock_guard guard(engine_lock);
    if (engine) return reply(request, "ok", nullptr);
    try {
        ninfer::EngineOptions options;
        options.artifact_path = path;
        options.device = 0;
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.suspend.enabled = true;
        engine = std::make_unique<ninfer::Engine>(std::move(options));
        reply(request, "ok", nullptr);
    } catch (const std::exception& error) {
        reply_error(request, error.what());
    }
}

void do_unload(const Request& request) {
    std::lock_guard guard(engine_lock);
    engine.reset();
    reply(request, "ok", nullptr);
}

void do_chat(const Request& request) {
    std::lock_guard guard(engine_lock);
    if (!engine) return reply_error(request, "not loaded");
    try {
        ninfer::PromptInput input;
        for (const auto& [role, text] : request.messages) {
            ninfer::ChatMessage message;
            message.role = role;
            message.parts.push_back({ninfer::MessagePartKind::Text, text, {}});
            input.messages.push_back(std::move(message));
        }
        input.options.enable_thinking = false;
        ninfer::RequestOptions options;
        options.execution.requested_output_tokens = 512;
        cancel_requested = false;
        DeltaSink sink(request);
        auto result = engine->generate(engine->prepare(std::move(input)), options, &sink,
                                       ninfer::CancellationView([] { return cancel_requested.load(); }));
        reply(request, "ok", &result.content);
    } catch (const std::exception& error) {
        reply_error(request, error.what());
    }
}

void run(Request request) {
    if (request.command == "status") {
        reply_state(request, engine ? "loaded" : "unloaded");
    } else if (request.command == "load") {
        do_load(request);
    } else if (request.command == "unload") {
        do_unload(request);
    } else if (request.command == "chat") {
        do_chat(request);
    } else if (request.command == "cancel") {
        cancel_requested = true;
        reply(request, "ok", nullptr);
    } else {
        reply_error(request, "unknown command");
    }
}

ninfer::ChatRole role_of(const char* atom) {
    if (std::strcmp(atom, "system") == 0) return ninfer::ChatRole::System;
    if (std::strcmp(atom, "assistant") == 0) return ninfer::ChatRole::Assistant;
    return ninfer::ChatRole::User;
}

std::string decode_binary(const char* buffer, int* index) {
    int type, size;
    ei_get_type(buffer, index, &type, &size);
    std::string text(static_cast<size_t>(size), '\0');
    long length = 0;
    if (type == ERL_BINARY_EXT) ei_decode_binary(buffer, index, text.data(), &length);
    text.resize(static_cast<size_t>(length));
    return text;
}

void decode_messages(const char* buffer, int* index, Request& request) {
    int count = 0;
    if (ei_decode_list_header(buffer, index, &count) != 0) return;
    for (int item = 0; item < count; item++) {
        int arity;
        char role[MAXATOMLEN];
        ei_decode_tuple_header(buffer, index, &arity);
        ei_decode_atom(buffer, index, role);
        request.messages.emplace_back(role_of(role), decode_binary(buffer, index));
    }
}

bool parse(int fd, const erlang_msg& message, ei_x_buff& x, Request& request) {
    int version, index = 0, arity, type, size;
    char command[MAXATOMLEN];
    if (ei_decode_version(x.buff, &index, &version) != 0 || ei_decode_tuple_header(x.buff, &index, &arity) != 0 ||
        arity != 3 || ei_decode_atom(x.buff, &index, command) != 0)
        return false;
    const int start = index;
    ei_skip_term(x.buff, &index);
    request.fd = fd;
    request.message = message;
    request.command = command;
    request.reference.assign(x.buff + start, static_cast<size_t>(index - start));
    ei_get_type(x.buff, &index, &type, &size);
    if (type == ERL_BINARY_EXT) request.text = decode_binary(x.buff, &index);
    else if (type == ERL_LIST_EXT) decode_messages(x.buff, &index, request);
    return true;
}

void serve_connection(int fd) {
    erlang_msg message;
    ei_x_buff x;
    ei_x_new(&x);
    for (;;) {
        x.index = 0;
        const int result = ei_xreceive_msg(fd, &message, &x);
        if (result == ERL_ERROR) break;
        Request request;
        if (result == ERL_MSG && (message.msgtype == ERL_SEND || message.msgtype == ERL_REG_SEND) &&
            parse(fd, message, x, request))
            std::thread(run, std::move(request)).detach();
    }
    ei_x_free(&x);
}

int listen_loopback(int* port) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    inet_pton(AF_INET, kHost, &address.sin_addr);
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    socklen_t length = sizeof address;
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 || listen(fd, 5) != 0) return -1;
    getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
    *port = ntohs(address.sin_port);
    return fd;
}

} // namespace

int main(int argc, char** argv) {
    const char* cookie = argc > 1 ? argv[1] : std::getenv("BONSAI_CNODE_COOKIE");
    if (cookie == nullptr) {
        std::fprintf(stderr, "usage: ninfer-cnode COOKIE\n");
        return 2;
    }
    in_addr loopback{};
    inet_pton(AF_INET, kHost, &loopback);
    ei_cnode node;
    int port = 0;
    ei_init();
    if (ei_connect_xinit(&node, kHost, "bonsai", "bonsai@127.0.0.1", &loopback, cookie, 0) < 0) return 1;
    const int listener = listen_loopback(&port);
    if (listener < 0 || ei_publish(&node, port) == -1) {
        std::fprintf(stderr, "cannot publish (is epmd running?)\n");
        return 1;
    }
    for (;;) {
        ErlConnect peer;
        const int fd = ei_accept(&node, listener, &peer);
        if (fd < 0) continue;
        serve_connection(fd);
        close(fd);
    }
}
