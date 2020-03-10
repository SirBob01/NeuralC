#ifndef HASH_H
#define HASH_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "list.h"

typedef struct {
    node_t **buckets;
    int size;
} hashmap_t;


hashmap_t *create_hashmap(int size);

void destroy_hashmap(hashmap_t *hashmap);

void add_pair(hashmap_t *hashmap, char key[], int value);

int get_value(hashmap_t *hashmap, char key[]);

long long int hash(char str[]);

#endif