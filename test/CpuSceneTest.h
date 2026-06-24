#pragma once

#include <gtest/gtest.h>
#include <hiprt/hiprt.h>
#include <vector>

class CpuSceneTest : public ::testing::Test
{
  protected:
	void SetUp() override;
	void TearDown() override;

	hiprtGeometry buildTriangleGeometry(
		std::vector<hiprtFloat3>& vertices,
		std::vector<uint32_t>&    indices );

	hiprtContext m_context = nullptr;
};
