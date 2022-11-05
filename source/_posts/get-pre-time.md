---
title: C语言如何获得精确时间
tags:
  - c
  - time
date: 2018-08-01 11:45:17
---


# C语言如何获得精确时间

## 精确到秒

```c
#include <stdio.h>
#include <time.h>  

int main(){
    time_t t_start, t_end;
    t_start = time(NULL) ;
    sleep(3000);
    t_end = time(NULL) ;
    printf("time: %.0f s\n", difftime(t_end,t_start)) ;
    return 0;
}
```

<!-- more -->

## 精确到微秒

```c
#include <stdio.h>
#include <sys/timeb.h>
 
long long getSystemTime() {
    struct timeb t;
    ftime(&t);
    return 1000 * t.time + t.millitm;
}
 
int main() {
    long long start=getSystemTime();
    sleep(3);
    long long end=getSystemTime();
 
    printf("time: %lld ms\n", end-start);
    return 0;
}
```
