#include <stdio.h>
#include <time.h>
#include <assert.h>

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

    long t = get_nanoseconds();
    printf("Duree approx entre deux instructions: %ld\n", get_nanoseconds() - t);

    {
        double probas[] = {0.25, 0.25, 0.25, 0.25};
        update_policy_probability(4, &probas, 0, 1);
        for (int i = 0; i < 4; ++i) {
            fprintf(stderr, "%lf ", probas[i]);
        }
        fputs("\n", stderr);
        assert(ial_same(probas[0], 1));
        assert(ial_same(probas[1], 0));
        assert(ial_same(probas[2], 0));
        assert(ial_same(probas[3], 0));
    }

    {
        double probas[] = {0.25, 0.25, 0.25, 0.25};
        update_policy_probability(4, &probas, 0, 0);
        for (int i = 0; i < 4; ++i) {
            fprintf(stderr, "%lf ", probas[i]);
        }
        fputs("\n", stderr);

        assert(ial_same(probas[0], 0));
        assert(ial_same(probas[1], 1/3.));
        assert(ial_same(probas[2], 1/3.));
        assert(ial_same(probas[3], 1/3.));
    }
}