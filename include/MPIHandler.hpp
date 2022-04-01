#pragma once
#include <tuple>
#include <optional>
#include <exception>

class MPIHandler {
    std::optional<std::pair<int, char**>> args;
    int inited;
    int finalized;

    MPIHandler() = default;
    ~MPIHandler();
public:
    struct NotInited : public std::exception {
        [[nodiscard]] const char* what() const noexcept override;
    };

    MPIHandler(const MPIHandler&) = delete;
    MPIHandler(MPIHandler&&) = delete;
    MPIHandler operator=(const MPIHandler&) = delete;
    MPIHandler operator=(MPIHandler&&) = delete;

    [[nodiscard]] static MPIHandler* getInstance();
    void setArgs(int argc, char** argv);
    void handleMPIResource();
};