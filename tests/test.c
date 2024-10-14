#include <stdio.h>
#include <time.h>
#include "ialib.h"

int main()
{
    puts(ial_version());
    struct timespec ts;
    clock_getres(CLOCK_REALTIME, &ts);
    printf("Précision de l'horloge realtime: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_MONOTONIC, &ts);
    printf("Précision de l'horloge monotonic: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_PROCESS_CPUTIME_ID, &ts);
    printf("Précision de l'horloge process: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_THREAD_CPUTIME_ID, &ts);
    printf("Précision de l'horloge thread: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_MONOTONIC_RAW, &ts);
    printf("Précision de l'horloge monotonic raw: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_MONOTONIC_RAW_APPROX, &ts);
    printf("Précision de l'horloge monotonic raw approx: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_UPTIME_RAW, &ts);
    printf("Précision de l'horloge uptime raw: %ld\n", ts.tv_nsec);
    clock_getres(CLOCK_MONOTONIC_RAW_APPROX, &ts);
    printf("Précision de l'horloge uptime raw approx: %ld\n", ts.tv_nsec);

}