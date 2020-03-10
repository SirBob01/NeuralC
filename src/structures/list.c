#include "list.h"

node_t *create_node(long long int key, int value) {
    node_t *node = (node_t *)malloc(sizeof(node_t));
    node->key = key;
    node->value = value;
    node->next = NULL;
    return node;
}

node_t *get_tail(node_t *node) {
    node_t *this = node;
    while(this) {
        this = this->next;
    }
    return this;
}

node_t *push_node(node_t *root, long long int key, int value) {
    node_t *tail = get_tail(root);
    tail->next = create_node(key, value);
    return tail->next;
}

void destroy_nodes(node_t *node) {
    node_t *this = node;
    node_t *next;
    while(this) {
        next = this->next;
        free(this);
        this = next;
    }
}