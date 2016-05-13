/*
 * Size.h
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#ifndef __SIZE_H__
#define __SIZE_H__
#include <iostream>

struct Size {

public:
	int w,h,wh;
	Size(int w=0, int h=0);
	Size(const Size& s);

	virtual ~Size();

	Size& operator=(const Size&);

	bool operator==(const Size&);
	bool operator!=(const Size&);

};
extern std::ostream& operator<<(std::ostream&,Size&);

#endif /* SIZE_H_ */
