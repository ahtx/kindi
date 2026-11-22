/* A small JSON reader for Kindi Monitor (RFC 8259 subset: no duplicate-key rules, numbers as double).
   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */
#include "json.h"
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *end; int depth; } Reader;

static void ws(Reader *r) {
    while (r->p < r->end && (*r->p == ' ' || *r->p == '\t' || *r->p == '\n' || *r->p == '\r')) r->p++;
}

static JNode *node(JType t) {
    JNode *n = (JNode *)calloc(1, sizeof *n);
    if (n) n->type = t;
    return n;
}

static int hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static int put_utf8(char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static char *string(Reader *r) {
    const char *s;
    char *out, *o;
    if (r->p >= r->end || *r->p != '"') return NULL;
    s = ++r->p;
    while (r->p < r->end && *r->p != '"') {
        if (*r->p == '\\') r->p++;
        r->p++;
    }
    if (r->p >= r->end) return NULL;
    out = o = (char *)malloc((size_t)(r->p - s) + 1);
    if (!out) return NULL;
    while (s < r->p) {
        if (*s != '\\') { *o++ = *s++; continue; }
        s++;
        switch (*s) {
        case 'n': *o++ = '\n'; s++; break;
        case 't': *o++ = '\t'; s++; break;
        case 'r': *o++ = '\r'; s++; break;
        case 'b': *o++ = '\b'; s++; break;
        case 'f': *o++ = '\f'; s++; break;
        case 'u': {
            unsigned cp = 0, lo = 0;
            if (r->p - s < 5 || !hex4(s + 1, &cp)) { free(out); return NULL; }
            s += 5;
            if (cp >= 0xD800 && cp < 0xDC00 && r->p - s >= 6 && s[0] == '\\' && s[1] == 'u' && hex4(s + 2, &lo) &&
                lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                s += 6;
            }
            o += put_utf8(o, cp);   /* never longer than the 6 escaped bytes it replaces */
            break;
        }
        default: *o++ = *s++; break;   /* \" \\ \/ */
        }
    }
    *o = 0;
    r->p++;
    return out;
}

static JNode *value(Reader *r);

static JNode *container(Reader *r, JType t) {
    JNode *n = node(t), *last = NULL;
    char close = t == J_OBJ ? '}' : ']';
    if (!n) return NULL;
    if (++r->depth > 64) { free(n); return NULL; }
    r->p++;
    ws(r);
    if (r->p < r->end && *r->p == close) { r->p++; r->depth--; return n; }
    for (;;) {
        char *key = NULL;
        JNode *v;
        ws(r);
        if (t == J_OBJ) {
            key = string(r);
            ws(r);
            if (!key || r->p >= r->end || *r->p != ':') { free(key); json_free(n); return NULL; }
            r->p++;
        }
        v = value(r);
        if (!v) { free(key); json_free(n); return NULL; }
        v->key = key;
        if (last) last->next = v; else n->child = v;
        last = v;
        n->count++;
        ws(r);
        if (r->p < r->end && *r->p == ',') { r->p++; continue; }
        if (r->p < r->end && *r->p == close) { r->p++; r->depth--; return n; }
        json_free(n);
        return NULL;
    }
}

static int word(Reader *r, const char *w) {
    size_t n = strlen(w);
    if ((size_t)(r->end - r->p) < n || memcmp(r->p, w, n) != 0) return 0;
    r->p += n;
    return 1;
}

static JNode *value(Reader *r) {
    JNode *n;
    ws(r);
    if (r->p >= r->end) return NULL;
    switch (*r->p) {
    case '{': return container(r, J_OBJ);
    case '[': return container(r, J_ARR);
    case '"':
        n = node(J_STR);
        if (n && !(n->str = string(r))) { free(n); return NULL; }
        return n;
    case 't': if (!word(r, "true")) return NULL; n = node(J_BOOL); if (n) n->num = 1; return n;
    case 'f': if (!word(r, "false")) return NULL; return node(J_BOOL);
    case 'n': if (!word(r, "null")) return NULL; return node(J_NULL);
    default: {
        char buf[64];
        size_t k = 0;
        while (r->p < r->end && k < sizeof buf - 1 && strchr("+-0123456789.eE", *r->p)) buf[k++] = *r->p++;
        if (!k) return NULL;
        buf[k] = 0;
        n = node(J_NUM);
        if (n) n->num = strtod(buf, NULL);
        return n;
    }
    }
}

JNode *json_parse(const char *text, int len) {
    Reader r;
    JNode *root;
    if (!text || len <= 0) return NULL;
    r.p = text;
    r.end = text + len;
    r.depth = 0;
    root = value(&r);
    return root;
}

void json_free(JNode *n) {
    while (n) {
        JNode *next = n->next;
        json_free(n->child);
        free(n->str);
        free(n->key);
        free(n);
        n = next;
    }
}

JNode *json_get(JNode *obj, const char *key) {
    JNode *c;
    if (!obj || obj->type != J_OBJ || !key) return NULL;
    for (c = obj->child; c; c = c->next)
        if (c->key && strcmp(c->key, key) == 0) return c;
    return NULL;
}

JNode *json_at(JNode *arr, int index) {
    JNode *c;
    if (!arr || (arr->type != J_ARR && arr->type != J_OBJ)) return NULL;
    for (c = arr->child; c && index > 0; c = c->next) index--;
    return c;
}

JNode *json_path(JNode *root, const char *path) {
    char part[64];
    const char *p = path;
    JNode *n = root;
    while (n && p && *p) {
        size_t k = 0;
        while (*p && *p != '.' && k < sizeof part - 1) part[k++] = *p++;
        part[k] = 0;
        if (*p == '.') p++;
        n = json_get(n, part);
    }
    return n;
}

double json_num(JNode *root, const char *path, double dflt) {
    JNode *n = json_path(root, path);
    return n && (n->type == J_NUM || n->type == J_BOOL) ? n->num : dflt;
}

const char *json_str(JNode *root, const char *path, const char *dflt) {
    JNode *n = json_path(root, path);
    return n && n->type == J_STR ? n->str : dflt;
}

int json_is(JNode *root, const char *path, JType type) {
    JNode *n = json_path(root, path);
    return n && n->type == type;
}
