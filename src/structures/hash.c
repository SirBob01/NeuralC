#include "hash.h"

hashmap_t *create_hashmap(int size) {
    hashmap_t *hashmap = (hashmap_t *)malloc(sizeof(hashmap_t));
    hashmap->buckets = (node_t **)malloc(sizeof(node_t *)*size);
    memset(hashmap->buckets, 0, sizeof(node_t *) * size);

    hashmap->size = size;
    return hashmap;
}

void destroy_hashmap(hashmap_t *hashmap) {
    int i;
    for(i = 0; i < hashmap->size; i++) {
        destroy_nodes(hashmap->buckets[i]);
    }
    free(hashmap->buckets);
    free(hashmap);
}

void add_pair(hashmap_t *hashmap, char key[], int value) {
    int hash_val = hash(key);
    int index = hash_val % hashmap->size;
    node_t *target = hashmap->buckets[index];
    if(!target) {
        hashmap->buckets[index] = create_node(hash_val, value);
    }
    else {
        push_node(hashmap->buckets[index], hash_val, value);
    }
}

int get_value(hashmap_t *hashmap, char key[]) {
    int hash_val = hash(key);
    int index = hash_val % hashmap->size;
    node_t *this = hashmap->buckets[index];
    while(this) {
        if(this->key == hash_val) {
            return this->value;
        }
        this = this->next;
    }
    exit(EXIT_FAILURE); // Key does not exist
}

long long int hash(char str[]) {
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