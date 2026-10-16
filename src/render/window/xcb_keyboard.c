#include "render/window/xcb_keyboard.h"

#include <xcb/xkb.h>
#include <xkbcommon/xkbcommon-compose.h>
#include <xkbcommon/xkbcommon-x11.h>

#include <stdio.h>
#include <stdlib.h>

int NukaXcbAutoRepeatPair(const void* release_raw, const void* press_raw) {
    if (!release_raw || !press_raw) return 0;
    const xcb_key_release_event_t* release = release_raw;
    const xcb_key_press_event_t* press = press_raw;
    return (release->response_type & 0x7fu) == XCB_KEY_RELEASE &&
           (press->response_type & 0x7fu) == XCB_KEY_PRESS &&
           release->detail == press->detail && release->time == press->time;
}

struct NukaXkbText {
    struct xkb_context* context;
    struct xkb_compose_table* table;
    struct xkb_compose_state* compose;
    char* buffer;
};

struct NukaXkbText* NukaXkbTextCreate(const char* locale) {
    struct NukaXkbText* text = calloc(1, sizeof(*text));
    if (!text) return NULL;
    text->context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!text->context) { free(text); return NULL; }
    text->table = xkb_compose_table_new_from_locale(text->context, locale, XKB_COMPOSE_COMPILE_NO_FLAGS);
    if (text->table) text->compose = xkb_compose_state_new(text->table, XKB_COMPOSE_STATE_NO_FLAGS);
    return text;
}

void NukaXkbTextDestroy(struct NukaXkbText* text) {
    if (!text) return;
    xkb_compose_state_unref(text->compose);
    xkb_compose_table_unref(text->table);
    xkb_context_unref(text->context);
    free(text->buffer);
    free(text);
}

void NukaXkbTextReset(struct NukaXkbText* text) {
    if (text && text->compose) xkb_compose_state_reset(text->compose);
}

const char* NukaXkbTextFeed(struct NukaXkbText* text, uint32_t keysym) {
    if (!text) return "";
    free(text->buffer);
    text->buffer = NULL;
    if (text->compose) {
        xkb_compose_state_feed(text->compose, keysym);
        switch (xkb_compose_state_get_status(text->compose)) {
            case XKB_COMPOSE_COMPOSING: return "";
            case XKB_COMPOSE_CANCELLED:
                xkb_compose_state_reset(text->compose);
                return "";
            case XKB_COMPOSE_COMPOSED: {
                const int length = xkb_compose_state_get_utf8(text->compose, NULL, 0);
                if (length > 0) {
                    text->buffer = malloc((size_t)length + 1u);
                    if (text->buffer) xkb_compose_state_get_utf8(text->compose, text->buffer, (size_t)length + 1u);
                }
                xkb_compose_state_reset(text->compose);
                return text->buffer ? text->buffer : "";
            }
            case XKB_COMPOSE_NOTHING: break;
        }
    }
    const uint32_t codepoint = xkb_keysym_to_utf32(keysym);
    if (codepoint < 0x20u || (codepoint >= 0x7fu && codepoint < 0xa0u)) return "";
    text->buffer = calloc(8u, 1u);
    if (text->buffer) xkb_keysym_to_utf8(keysym, text->buffer, 8u);
    return text->buffer ? text->buffer : "";
}

struct NukaXcbKeyboard {
    xcb_connection_t* connection;
    struct xkb_context* context;
    struct xkb_keymap* keymap;
    struct xkb_state* state;
    struct NukaXkbText* text;
    int32_t device;
    uint8_t event_base;
};

static int RebuildKeymap(struct NukaXcbKeyboard* keyboard) {
    struct xkb_keymap* keymap = xkb_x11_keymap_new_from_device(
        keyboard->context, keyboard->connection, keyboard->device, XKB_KEYMAP_COMPILE_NO_FLAGS);
    struct xkb_state* state = keymap ? xkb_x11_state_new_from_device(keymap, keyboard->connection, keyboard->device) : NULL;
    xkb_state_unref(keyboard->state);
    xkb_keymap_unref(keyboard->keymap);
    keyboard->state = state;
    keyboard->keymap = keymap;
    NukaXkbTextReset(keyboard->text);
    return state != NULL;
}

struct NukaXcbKeyboard* NukaXcbKeyboardCreate(xcb_connection_t* connection) {
    struct NukaXcbKeyboard* keyboard = calloc(1, sizeof(*keyboard));
    if (!keyboard) return NULL;
    keyboard->connection = connection;
    keyboard->context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!keyboard->context || !xkb_x11_setup_xkb_extension(connection,
            XKB_X11_MIN_MAJOR_XKB_VERSION, XKB_X11_MIN_MINOR_XKB_VERSION,
            XKB_X11_SETUP_XKB_EXTENSION_NO_FLAGS, NULL, NULL, &keyboard->event_base, NULL)) {
        NukaXcbKeyboardDestroy(keyboard);
        return NULL;
    }
    keyboard->device = xkb_x11_get_core_keyboard_device_id(connection);
    if (keyboard->device < 0 || !RebuildKeymap(keyboard)) {
        NukaXcbKeyboardDestroy(keyboard);
        return NULL;
    }
    const uint32_t repeat = XCB_XKB_PER_CLIENT_FLAG_DETECTABLE_AUTO_REPEAT;
    const xcb_xkb_per_client_flags_cookie_t repeat_cookie = xcb_xkb_per_client_flags(
        connection, (xcb_xkb_device_spec_t)keyboard->device, repeat, repeat, 0, 0, 0);
    xcb_xkb_per_client_flags_reply_t* repeat_reply = xcb_xkb_per_client_flags_reply(connection, repeat_cookie, NULL);
    if (!repeat_reply || (repeat_reply->supported & repeat) == 0u || (repeat_reply->value & repeat) == 0u)
        fprintf(stderr, "[nuka_window_xcb] filtering legacy keyboard autorepeat pairs\n");
    free(repeat_reply);
    const uint16_t events = XCB_XKB_EVENT_TYPE_NEW_KEYBOARD_NOTIFY |
                            XCB_XKB_EVENT_TYPE_MAP_NOTIFY | XCB_XKB_EVENT_TYPE_STATE_NOTIFY;
    const uint16_t maps = XCB_XKB_MAP_PART_KEY_TYPES | XCB_XKB_MAP_PART_KEY_SYMS |
                         XCB_XKB_MAP_PART_MODIFIER_MAP | XCB_XKB_MAP_PART_EXPLICIT_COMPONENTS |
                         XCB_XKB_MAP_PART_KEY_ACTIONS | XCB_XKB_MAP_PART_KEY_BEHAVIORS |
                         XCB_XKB_MAP_PART_VIRTUAL_MODS | XCB_XKB_MAP_PART_VIRTUAL_MOD_MAP;
    xcb_void_cookie_t cookie = xcb_xkb_select_events_checked(connection, (xcb_xkb_device_spec_t)keyboard->device,
                                                            events, 0, events, maps, maps, NULL);
    xcb_generic_error_t* error = xcb_request_check(connection, cookie);
    if (error) { free(error); NukaXcbKeyboardDestroy(keyboard); return NULL; }
    const char* locale = getenv("LC_ALL");
    if (!locale || !*locale) locale = getenv("LC_CTYPE");
    if (!locale || !*locale) locale = getenv("LANG");
    if (!locale || !*locale) locale = "C";
    keyboard->text = NukaXkbTextCreate(locale);
    return keyboard;
}

void NukaXcbKeyboardDestroy(struct NukaXcbKeyboard* keyboard) {
    if (!keyboard) return;
    NukaXkbTextDestroy(keyboard->text);
    xkb_state_unref(keyboard->state);
    xkb_keymap_unref(keyboard->keymap);
    xkb_context_unref(keyboard->context);
    free(keyboard);
}

int NukaXcbKeyboardHandleEvent(struct NukaXcbKeyboard* keyboard, const void* raw) {
    if (!keyboard) return 0;
    const xcb_xkb_state_notify_event_t* event = raw;
    if ((event->response_type & 0x7fu) != keyboard->event_base) return 0;
    if (event->xkbType == XCB_XKB_STATE_NOTIFY && keyboard->state) {
        xkb_state_update_mask(keyboard->state, event->baseMods, event->latchedMods, event->lockedMods,
                              (xkb_layout_index_t)event->baseGroup, (xkb_layout_index_t)event->latchedGroup,
                              event->lockedGroup);
    } else if (event->xkbType == XCB_XKB_NEW_KEYBOARD_NOTIFY || event->xkbType == XCB_XKB_MAP_NOTIFY) {
        keyboard->device = xkb_x11_get_core_keyboard_device_id(keyboard->connection);
        if (keyboard->device < 0) {
            xkb_state_unref(keyboard->state);
            keyboard->state = NULL;
            NukaXkbTextReset(keyboard->text);
        }
        if (keyboard->device < 0 || !RebuildKeymap(keyboard))
            fprintf(stderr, "[nuka_window_xcb] keyboard map unavailable\n");
    }
    return 1;
}

uint32_t NukaXcbKeyboardKeysym(struct NukaXcbKeyboard* keyboard, uint8_t keycode) {
    if (!keyboard || !keyboard->state) return 0u;
    const xkb_keysym_t* symbols = NULL;
    const xkb_layout_index_t layout = xkb_state_key_get_layout(keyboard->state, keycode);
    const int count = xkb_keymap_key_get_syms_by_level(keyboard->keymap, keycode, layout, 0, &symbols);
    return count > 0 ? symbols[0] : 0u;
}

const char* NukaXcbKeyboardText(struct NukaXcbKeyboard* keyboard, uint8_t keycode) {
    if (!keyboard || !keyboard->state) return "";
    if (xkb_state_mod_name_is_active(keyboard->state, XKB_MOD_NAME_CTRL, XKB_STATE_MODS_EFFECTIVE) > 0 ||
        xkb_state_mod_name_is_active(keyboard->state, XKB_MOD_NAME_ALT, XKB_STATE_MODS_EFFECTIVE) > 0 ||
        xkb_state_mod_name_is_active(keyboard->state, XKB_MOD_NAME_LOGO, XKB_STATE_MODS_EFFECTIVE) > 0) {
        NukaXkbTextReset(keyboard->text);
        return "";
    }
    return NukaXkbTextFeed(keyboard->text, xkb_state_key_get_one_sym(keyboard->state, keycode));
}

void NukaXcbKeyboardResetText(struct NukaXcbKeyboard* keyboard) {
    if (keyboard) NukaXkbTextReset(keyboard->text);
}
