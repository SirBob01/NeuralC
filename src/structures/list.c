#include "list.h"

node_t *node_create(long long int key, void *data, size_t unit) {
    node_t *node = (node_t *)malloc(sizeof(node_t));
    node->key = key;
    node->unit = unit;

    node->data = malloc(unit);
    memcpy(node->data, data, unit);

    node->next = NULL;
    return node;
}

node_t *node_get_tail(node_t *node) {
    node_t *this = node;
    while(this) {
        this = this->next;
    }
    return this;
}

node_t *node_push(node_t *root, long long int key, void *data) {
    node_t *tail = node_get_tail(root);
    tail->next = node_create(key, data, tail->unit);
    return tail->next;
}

void nodes_destroy(node_t *node) {
    node_t *this = node;
    node_t *next;
    while(this) {
        next = this->next;
        free(this->data);
        free(this);
        this = next;
    }
}