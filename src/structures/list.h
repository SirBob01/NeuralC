#ifndef LIST_H
#define LIST_H

#include <stdlib.h>
#include <string.h>

typedef struct node {
    long long int key;
    size_t unit; // Size of data to be stored

    void *data;
    struct node *next;
} node_t;


node_t *node_create(long long int key, void *data, size_t unit);

node_t *node_get_tail(node_t *node);

node_t *node_push(node_t *root, long long int key, void *data);

void nodes_destroy(node_t *node);

#endif