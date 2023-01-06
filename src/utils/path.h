#pragma once
#include <iostream>
#include <direct.h>

inline void makedirs(const char* path) {
	if (strlen(path) > _MAX_PATH) {
		throw "path length over MAX_LENGTH";
	}

	int path_length = strlen(path);
	int leave_length = 0;
	int created_length = 0;
	char size_path_temp[_MAX_PATH] = { 0 };
	while (true) {
		int pos = -1;
		if (NULL != strchr(path + created_length, '\\')) {
			pos = strlen(strchr(path + created_length, '\\')) - 1;
		}
		else if (NULL != strchr(path + created_length, '/')) {
			pos = strlen(strchr(path + created_length, '/')) - 1;
		}
		else {
			break;
		}
		leave_length = pos;
		created_length = path_length - leave_length;
		strncpy(size_path_temp, path, created_length);
		mkdir(size_path_temp);
	}
	if (created_length < path_length) {
		mkdir(path);
	}
}

