#include "hash.h"

hashmap_t *hashmap_create(int size, size_t unit) {
    hashmap_t *hashmap = (hashmap_t *)malloc(sizeof(hashmap_t));
    hashmap->buckets = (node_t **)malloc(sizeof(node_t *)*size);
    memset(hashmap->buckets, 0, sizeof(node_t *) * size);

    hashmap->unit = unit;
    hashmap->size = size;
    return hashmap;
}

void hashmap_destroy(hashmap_t *hashmap) {
    int i;
    for(i = 0; i < hashmap->size; i++) {
        nodes_destroy(hashmap->buckets[i]);
    }
    free(hashmap->buckets);
    free(hashmap);
}

void hashmap_append(hashmap_t *hashmap, const char *key, void *data) {
    int hash_val = hash(key);
    int index = hash_val % hashmap->size;
    node_t *target = hashmap->buckets[index];
    if(!target) {
        hashmap->buckets[index] = node_create(
            hash_val, 
            data, 
            hashmap->unit
        );
    }
    else {
        node_push(hashmap->buckets[index], hash_val, data);
    }
}

node_t *hashmap_get(hashmap_t *hashmap, const char *key) {
    int hash_val = hash(key);
    int index = hash_val % hashmap->size;
    node_t *this = hashmap->buckets[index];
    while(this) {
        if(this->key == hash_val) {
            return this;
        }
        this = this->next;
    }
    return NULL;
}

long long int hash(const char *str) {
    long long int hash = 0;
    long long int m = pow(10, 9) + 9;
    long long int p_pow = 1;
    long long int p = pow(2, 13)-1;
    int len = strlen(str);
    
    int i;
    for(i = 0; i < len; i++) {
        hash += ((str[i]-'a'+1) * p_pow);
        p_pow *= p;
    }
    return hash % m;
}