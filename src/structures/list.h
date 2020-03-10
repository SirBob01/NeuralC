#ifndef LIST_H
#define LIST_H

#include <stdlib.h>

typedef struct node {
    long long int key;
    int value;
    struct node *next;
} node_t;


node_t *create_node(long long int key, int value);

node_t *get_tail(node_t *node);

node_t *push_node(node_t *root, long long int key, int value);

void destroy_nodes(node_t *node);

#endif