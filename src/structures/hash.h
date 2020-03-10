#ifndef HASH_H
#define HASH_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "list.h"

typedef struct {
    node_t **buckets;
    size_t unit;
    int size;
} hashmap_t;


hashmap_t *hashmap_create(int size, size_t unit);

void hashmap_destroy(hashmap_t *hashmap);

void hashmap_append(hashmap_t *hashmap, char key[], void *data);

node_t *hashmap_get(hashmap_t *hashmap, char key[]);

// Utility function to hash a string
long long int hash(char str[]);

#endif