/* A small JSON reader for Kindi Monitor: parse a document into a tree, then look values up by path.
   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */
#ifndef KINDI_JSON_H
#define KINDI_JSON_H

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JNode {
    JType type;
    double num;              /* J_NUM, J_BOOL (0 or 1) */
    char *str;               /* J_STR: UTF-8, unescaped */
    char *key;               /* the member name when the parent is an object */
    struct JNode *child;     /* J_ARR, J_OBJ: the first element */
    struct JNode *next;      /* the next element of the parent */
    int count;               /* J_ARR, J_OBJ: the number of elements */
} JNode;

/* parse text (len bytes); returns the root or NULL. Free with json_free. */
JNode *json_parse(const char *text, int len);
void json_free(JNode *root);

/* member of an object by name, element of an array by index; NULL when absent */
JNode *json_get(JNode *obj, const char *key);
JNode *json_at(JNode *arr, int index);
/* a path of member names separated by dots, e.g. "busy.phase" */
JNode *json_path(JNode *root, const char *path);

/* typed reads with a default */
double json_num(JNode *root, const char *path, double dflt);
const char *json_str(JNode *root, const char *path, const char *dflt);
int json_is(JNode *root, const char *path, JType type);

#endif
