/*
 * Size.cpp
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#include "Size.h"

Size::Size(int w, int h):w(w),h(h),wh(w*h){
	// TODO Auto-generated constructor stub

}

Size::Size(const Size& s):w(s.w),h(s.h),wh(s.wh){

}

Size::~Size() {
	// TODO Auto-generated destructor stub
}

Size& Size::operator=(const Size& s){
	w = s.w;
	h = s.h;
	wh = s.wh;
	return *this;
}
bool Size::operator!=(const Size& s){
	return !(*this == s);
}
bool Size::operator==(const Size& s){
	return (w == s.w) && (h == s.h);
}
