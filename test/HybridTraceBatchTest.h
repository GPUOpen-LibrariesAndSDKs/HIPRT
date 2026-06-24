#pragma once

#include "HybridSceneTest.h"

class HybridTraceBatchTest : public HybridSceneTest
{
  protected:
	hiprtScene buildSingleInstanceScene( hiprtGeometry& geomOut );
};
