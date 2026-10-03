#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct xcb_connection_t;
struct NukaXcbKeyboard;
struct NukaXkbText;

int NukaXcbAutoRepeatPair(const void* release, const void* press);

struct NukaXkbText* NukaXkbTextCreate(const char* locale);
void NukaXkbTextDestroy(struct NukaXkbText* text);
const char* NukaXkbTextFeed(struct NukaXkbText* text, uint32_t keysym);
void NukaXkbTextReset(struct NukaXkbText* text);

struct NukaXcbKeyboard* NukaXcbKeyboardCreate(struct xcb_connection_t* connection);
void NukaXcbKeyboardDestroy(struct NukaXcbKeyboard* keyboard);
int NukaXcbKeyboardHandleEvent(struct NukaXcbKeyboard* keyboard, const void* event);
uint32_t NukaXcbKeyboardKeysym(struct NukaXcbKeyboard* keyboard, uint8_t keycode);
const char* NukaXcbKeyboardText(struct NukaXcbKeyboard* keyboard, uint8_t keycode);
void NukaXcbKeyboardResetText(struct NukaXcbKeyboard* keyboard);

#ifdef __cplusplus
}
#endif
