#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    int thread_count;
    if (argv[1] == NULL) {
        thread_count = 4;
    }
    else {
        thread_count = strtol(argv[1], NULL, 10);
    }
#pragma omp parallel num_threads(thread_count)
    {
        printf("Hello from thread %d of %d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
    }

    return 0;
}