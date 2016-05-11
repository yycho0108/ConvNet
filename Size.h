/*
 * Size.h
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#ifndef __SIZE_H__
#define __SIZE_H__

struct Size {
	int w,h,wh;
public:
	Size(int w=0, int h=0);
	Size(const Size& s);

	virtual ~Size();

	Size& operator=(const Size&);

	bool operator==(Size&);
	bool operator!=(Size&);

};

#endif /* SIZE_H_ */
