 

import pandas as pd



mylist = [['1/1/2018', 27], ['1/2/2018', 24], ['1/3/2018', 26], ['1/4/2018', 23], ['1/5/2018', 20], ['1/6/2018', 23], ['1/7/2018', 27], ['1/8/2018', 23], ['1/9/2018', 24], ['1/10/2018', 28], ['1/11/2018', 28], ['1/12/2018', 22], ['1/13/2018', 23], ['1/14/2018', 30], ['1/15/2018', 30], ['1/16/2018', 22], ['1/17/2018', 21], ['1/18/2018', 28], ['1/19/2018', 26], ['1/20/2018', 30], ['1/21/2018', 20], ['1/22/2018', 20], ['1/23/2018', 23], ['1/24/2018', 25], ['1/25/2018', 28], ['1/26/2018', 26], ['1/27/2018', 29], ['1/28/2018', 27], ['1/29/2018', 21], ['1/30/2018', 25], ['1/31/2018', 25], ['2/1/2018', 23], ['2/2/2018', 30], ['2/3/2018', 24], ['2/4/2018', 28], ['2/5/2018', 26], ['2/6/2018', 20], ['2/7/2018', 27], ['2/8/2018', 23], ['2/9/2018', 25], ['2/10/2018', 30], ['2/11/2018', 20], ['2/12/2018', 21], ['2/13/2018', 26], ['2/14/2018', 30], ['2/15/2018', 24], ['2/16/2018', 21], ['2/17/2018', 26], ['2/18/2018', 24], ['2/19/2018', 28], ['2/20/2018', 30], ['2/21/2018', 25], ['2/22/2018', 27], ['2/23/2018', 25], ['2/24/2018', 28], ['2/25/2018', 27], ['2/26/2018', 23], ['2/27/2018', 21], ['2/28/2018', 21], ['3/1/2018', 23], ['3/2/2018', 23], ['3/3/2018', 22], ['3/4/2018', 26], ['3/5/2018', 26], ['3/6/2018', 29], ['3/7/2018', 26], ['3/8/2018', 22], ['3/9/2018', 22], ['3/10/2018', 22], ['3/11/2018', 25], ['3/12/2018', 20], ['3/13/2018', 23], ['3/14/2018', 22], ['3/15/2018', 25], ['3/16/2018', 22], ['3/17/2018', 27], ['3/18/2018', 27], ['3/19/2018', 23], ['3/20/2018', 26], ['3/21/2018', 23], ['3/22/2018', 35], ['3/23/2018', 34], ['3/24/2018', 22], ['3/25/2018', 33], ['3/26/2018', 28], ['3/27/2018', 31], ['3/28/2018', 23], ['3/29/2018', 37], ['3/30/2018', 25], ['3/31/2018', 19], ['4/1/2018', 28], ['4/2/2018', 25], ['4/3/2018', 24], ['4/4/2018', 30], ['4/5/2018', 24], ['4/6/2018', 26], ['4/7/2018', 26], ['4/8/2018', 23], ['4/9/2018', 38], ['4/10/2018', 34], ['4/11/2018', 26], ['4/12/2018', 35], ['4/13/2018', 23], ['4/14/2018', 36], ['4/15/2018', 24], ['4/16/2018', 32], ['4/17/2018', 21], ['4/18/2018', 36], ['4/19/2018', 33], ['4/20/2018', 33], ['4/21/2018', 31], ['4/22/2018', 33], ['4/23/2018', 23], ['4/24/2018', 25], ['4/25/2018', 32], ['4/26/2018', 35], ['4/27/2018', 24], ['4/28/2018', 20], ['4/29/2018', 25], ['4/30/2018', 33], ['5/1/2018', 23], ['5/2/2018', 34], ['5/3/2018', 24], ['5/4/2018', 35], ['5/5/2018', 32], ['5/6/2018', 36], ['5/7/2018', 34], ['5/8/2018', 34], ['5/9/2018', 23], ['5/10/2018', 35], ['5/11/2018', 21], ['5/12/2018', 23], ['5/13/2018', 37], ['5/14/2018', 30], ['5/15/2018', 38], ['5/16/2018', 28], ['5/17/2018', 35], ['5/18/2018', 31], ['5/19/2018', 25], ['5/20/2018', 20], ['5/21/2018', 36], ['5/22/2018', 35], ['5/23/2018', 23], ['5/24/2018', 25], ['5/25/2018', 37], ['5/26/2018', 19], ['5/27/2018', 33], ['5/28/2018', 35], ['5/29/2018', 31], ['5/30/2018', 28], ['5/31/2018', 23], ['6/1/2018', 32], ['6/2/2018', 32], ['6/3/2018', 24], ['6/4/2018', 32], ['6/5/2018', 31], ['6/6/2018', 37], ['6/7/2018', 24], ['6/8/2018', 21], ['6/9/2018', 24], ['6/10/2018', 29], ['6/11/2018', 23], ['6/12/2018', 25], ['6/13/2018', 23], ['6/14/2018', 28], ['6/15/2018', 24], ['6/16/2018', 23], ['6/17/2018', 20], ['6/18/2018', 28], ['6/19/2018', 26], ['6/20/2018', 29], ['6/21/2018', 21], ['6/22/2018', 28], ['6/23/2018', 25], ['6/24/2018', 28], ['6/25/2018', 29], ['6/26/2018', 26], ['6/27/2018', 29], ['6/28/2018', 21], ['6/29/2018', 26], ['6/30/2018', 22], ['7/1/2018', 23], ['7/2/2018', 20], ['7/3/2018', 25], ['7/4/2018', 20], ['7/5/2018', 24], ['7/6/2018', 27], ['7/7/2018', 26], ['7/8/2018', 21], ['7/9/2018', 28], ['7/10/2018', 24], ['7/11/2018', 30], ['7/12/2018', 30], ['7/13/2018', 22], ['7/14/2018', 28], ['7/15/2018', 28], ['7/16/2018', 22], ['7/17/2018', 20], ['7/18/2018', 26], ['7/19/2018', 28], ['7/20/2018', 20], ['7/21/2018', 26], ['7/22/2018', 29], ['7/23/2018', 29], ['7/24/2018', 23], ['7/25/2018', 29], ['7/26/2018', 25], ['7/27/2018', 22], ['7/28/2018', 30], ['7/29/2018', 26], ['7/30/2018', 26], ['7/31/2018', 20], ['8/1/2018', 28], ['8/2/2018', 23], ['8/3/2018', 28], ['8/4/2018', 25], ['8/5/2018', 20], ['8/6/2018', 27], ['8/7/2018', 29], ['8/8/2018', 27], ['8/9/2018', 20], ['8/10/2018', 28], ['8/11/2018', 26], ['8/12/2018', 30], ['8/13/2018', 20], ['8/14/2018', 29], ['8/15/2018', 24], ['8/16/2018', 30], ['8/17/2018', 26], ['8/18/2018', 21], ['8/19/2018', 22], ['8/20/2018', 21], ['8/21/2018', 25], ['8/22/2018', 22], ['8/23/2018', 29], ['8/24/2018', 24], ['8/25/2018', 21], ['8/26/2018', 36], ['8/27/2018', 25], ['8/28/2018', 22], ['8/29/2018', 28], ['8/30/2018', 21], ['8/31/2018', 23], ['9/1/2018', 29], ['9/2/2018', 26], ['9/3/2018', 31], ['9/4/2018', 38], ['9/5/2018', 19], ['9/6/2018', 28], ['9/7/2018', 32], ['9/8/2018', 29], ['9/9/2018', 20], ['9/10/2018', 24], ['9/11/2018', 29], ['9/12/2018', 35], ['9/13/2018', 28], ['9/14/2018', 36], ['9/15/2018', 32], ['9/16/2018', 24], ['9/17/2018', 19], ['9/18/2018', 30], ['9/19/2018', 21], ['9/20/2018', 23], ['9/21/2018', 23], ['9/22/2018', 35], ['9/23/2018', 21], ['9/24/2018', 37], ['9/25/2018', 29], ['9/26/2018', 34], ['9/27/2018', 24], ['9/28/2018', 37], ['9/29/2018', 32], ['9/30/2018', 23], ['10/1/2018', 25], ['10/2/2018', 29], ['10/3/2018', 31], ['10/4/2018', 30], ['10/5/2018', 36], ['10/6/2018', 27], ['10/7/2018', 28], ['10/8/2018', 22], ['10/9/2018', 20], ['10/10/2018', 23], ['10/11/2018', 29], ['10/12/2018', 21], ['10/13/2018', 28], ['10/14/2018', 21], ['10/15/2018', 25], ['10/16/2018', 28], ['10/17/2018', 25], ['10/18/2018', 29], ['10/19/2018', 20], ['10/20/2018', 28], ['10/21/2018', 20], ['10/22/2018', 28], ['10/23/2018', 28], ['10/24/2018', 29], ['10/25/2018', 25], ['10/26/2018', 28], ['10/27/2018', 22], ['10/28/2018', 22], ['10/29/2018', 29], ['10/30/2018', 24], ['10/31/2018', 25], ['11/1/2018', 28], ['11/2/2018', 25], ['11/3/2018', 29], ['11/4/2018', 21], ['11/5/2018', 23], ['11/6/2018', 22], ['11/7/2018', 24], ['11/8/2018', 29], ['11/9/2018', 25], ['11/10/2018', 21], ['11/11/2018', 22], ['11/12/2018', 24], ['11/13/2018', 23], ['11/14/2018', 20], ['11/15/2018', 25], ['11/16/2018', 26], ['11/17/2018', 30], ['11/18/2018', 29], ['11/19/2018', 20], ['11/20/2018', 24], ['11/21/2018', 24], ['11/22/2018', 26], ['11/23/2018', 27], ['11/24/2018', 22], ['11/25/2018', 26], ['11/26/2018', 20], ['11/27/2018', 28], ['11/28/2018', 24], ['11/29/2018', 20], ['11/30/2018', 22], ['12/1/2018', 27], ['12/2/2018', 27], ['12/3/2018', 21], ['12/4/2018', 23], ['12/5/2018', 29], ['12/6/2018', 23], ['12/7/2018', 20], ['12/8/2018', 26], ['12/9/2018', 27], ['12/10/2018', 26], ['12/11/2018', 25], ['12/12/2018', 24], ['12/13/2018', 26], ['12/14/2018', 24], ['12/15/2018', 28], ['12/16/2018', 20], ['12/17/2018', 29], ['12/18/2018', 24], ['12/19/2018', 24], ['12/20/2018', 30], ['12/21/2018', 22], ['12/22/2018', 20], ['12/23/2018', 28], ['12/24/2018', 27], ['12/25/2018', 21], ['12/26/2018', 26], ['12/27/2018', 26], ['12/28/2018', 36], ['12/29/2018', 35], ['12/30/2018', 38], ['12/31/2018', 32], ['1/1/2019', 36], ['1/2/2019', 21], ['1/3/2019', 38], ['1/4/2019', 27], ['1/5/2019', 31], ['1/6/2019', 35], ['1/7/2019', 32], ['1/8/2019', 28], ['1/9/2019', 23], ['1/10/2019', 21], ['1/11/2019', 36], ['1/12/2019', 28], ['1/13/2019', 37], ['1/14/2019', 20], ['1/15/2019', 23], ['1/16/2019', 20], ['1/17/2019', 37], ['1/18/2019', 31], ['1/19/2019', 19], ['1/20/2019', 36], ['1/21/2019', 32], ['1/22/2019', 24], ['1/23/2019', 25], ['1/24/2019', 32], ['1/25/2019', 31], ['1/26/2019', 31], ['1/27/2019', 29], ['1/28/2019', 20], ['1/29/2019', 30], ['1/30/2019', 19], ['1/31/2019', 20], ['2/1/2019', 35], ['2/2/2019', 23], ['2/3/2019', 26], ['2/4/2019', 32], ['2/5/2019', 37], ['2/6/2019', 36], ['2/7/2019', 26], ['2/8/2019', 25], ['2/9/2019', 26], ['2/10/2019', 27], ['2/11/2019', 22], ['2/12/2019', 20], ['2/13/2019', 26], ['2/14/2019', 28], ['2/15/2019', 30], ['2/16/2019', 23], ['2/17/2019', 27], ['2/18/2019', 23], ['2/19/2019', 23], ['2/20/2019', 20], ['2/21/2019', 29], ['2/22/2019', 28], ['2/23/2019', 24], ['2/24/2019', 24], ['2/25/2019', 26], ['2/26/2019', 20], ['2/27/2019', 25], ['2/28/2019', 22], ['3/1/2019', 24], ['3/2/2019', 28], ['3/3/2019', 23], ['3/4/2019', 22], ['3/5/2019', 21], ['3/6/2019', 22], ['3/7/2019', 21], ['3/8/2019', 25], ['3/9/2019', 26], ['3/10/2019', 30], ['3/11/2019', 21], ['3/12/2019', 23], ['3/13/2019', 29], ['3/14/2019', 24], ['3/15/2019', 24], ['3/16/2019', 28], ['3/17/2019', 22], ['3/18/2019', 23], ['3/19/2019', 23], ['3/20/2019', 20], ['3/21/2019', 21], ['3/22/2019', 22], ['3/23/2019', 22], ['3/24/2019', 30], ['3/25/2019', 28], ['3/26/2019', 30], ['3/27/2019', 24], ['3/28/2019', 24], ['3/29/2019', 21], ['3/30/2019', 21], ['3/31/2019', 23], ['4/1/2019', 22], ['4/2/2019', 28], ['4/3/2019', 22], ['4/4/2019', 20], ['4/5/2019', 24], ['4/6/2019', 23], ['4/7/2019', 20], ['4/8/2019', 23], ['4/9/2019', 23], ['4/10/2019', 26], ['4/11/2019', 30], ['4/12/2019', 29], ['4/13/2019', 23], ['4/14/2019', 21], ['4/15/2019', 21], ['4/16/2019', 23], ['4/17/2019', 27], ['4/18/2019', 28], ['4/19/2019', 20], ['4/20/2019', 25], ['4/21/2019', 20], ['4/22/2019', 24], ['4/23/2019', 24], ['4/24/2019', 23], ['4/25/2019', 29], ['4/26/2019', 24], ['4/27/2019', 26], ['4/28/2019', 32], ['4/29/2019', 26], ['4/30/2019', 22], ['5/1/2019', 30], ['5/2/2019', 19], ['5/3/2019', 19], ['5/4/2019', 24], ['5/5/2019', 27], ['5/6/2019', 30], ['5/7/2019', 34], ['5/8/2019', 36], ['5/9/2019', 35], ['5/10/2019', 24], ['5/11/2019', 37], ['5/12/2019', 27], ['5/13/2019', 26], ['5/14/2019', 33], ['5/15/2019', 22], ['5/16/2019', 23], ['5/17/2019', 25], ['5/18/2019', 35], ['5/19/2019', 32], ['5/20/2019', 24], ['5/21/2019', 36], ['5/22/2019', 27], ['5/23/2019', 21], ['5/24/2019', 38], ['5/25/2019', 22], ['5/26/2019', 21], ['5/27/2019', 27], ['5/28/2019', 26], ['5/29/2019', 30], ['5/30/2019', 35], ['5/31/2019', 38], ['6/1/2019', 33], ['6/2/2019', 19], ['6/3/2019', 34], ['6/4/2019', 38], ['6/5/2019', 26], ['6/6/2019', 33], ['6/7/2019', 22], ['6/8/2019', 34], ['6/9/2019', 28], ['6/10/2019', 23], ['6/11/2019', 22], ['6/12/2019', 24], ['6/13/2019', 21], ['6/14/2019', 21], ['6/15/2019', 21], ['6/16/2019', 29], ['6/17/2019', 21], ['6/18/2019', 28], ['6/19/2019', 29], ['6/20/2019', 29], ['6/21/2019', 22], ['6/22/2019', 27], ['6/23/2019', 25], ['6/24/2019', 22], ['6/25/2019', 20], ['6/26/2019', 23], ['6/27/2019', 22], ['6/28/2019', 29], ['6/29/2019', 23], ['6/30/2019', 22], ['7/1/2019', 21], ['7/2/2019', 26], ['7/3/2019', 27], ['7/4/2019', 22], ['7/5/2019', 29], ['7/6/2019', 20], ['7/7/2019', 25], ['7/8/2019', 22], ['7/9/2019', 27], ['7/10/2019', 23], ['7/11/2019', 25], ['7/12/2019', 27], ['7/13/2019', 24], ['7/14/2019', 23], ['7/15/2019', 23], ['7/16/2019', 21], ['7/17/2019', 21], ['7/18/2019', 28], ['7/19/2019', 23], ['7/20/2019', 21], ['7/21/2019', 20], ['7/22/2019', 26], ['7/23/2019', 23], ['7/24/2019', 20], ['7/25/2019', 27], ['7/26/2019', 24], ['7/27/2019', 20], ['7/28/2019', 29], ['7/29/2019', 22], ['7/30/2019', 30], ['7/31/2019', 25], ['8/1/2019', 30], ['8/2/2019', 21], ['8/3/2019', 25], ['8/4/2019', 30], ['8/5/2019', 22], ['8/6/2019', 26], ['8/7/2019', 20], ['8/8/2019', 30], ['8/9/2019', 25], ['8/10/2019', 27], ['8/11/2019', 24], ['8/12/2019', 28], ['8/13/2019', 21], ['8/14/2019', 26], ['8/15/2019', 28], ['8/16/2019', 21], ['8/17/2019', 23], ['8/18/2019', 22], ['8/19/2019', 23], ['8/20/2019', 28], ['8/21/2019', 25], ['8/22/2019', 23], ['8/23/2019', 28], ['8/24/2019', 22], ['8/25/2019', 29], ['8/26/2019', 26], ['8/27/2019', 25], ['8/28/2019', 23], ['8/29/2019', 32], ['8/30/2019', 38], ['8/31/2019', 36], ['9/1/2019', 32], ['9/2/2019', 38], ['9/3/2019', 30], ['9/4/2019', 35], ['9/5/2019', 36], ['9/6/2019', 33], ['9/7/2019', 22], ['9/8/2019', 22], ['9/9/2019', 21], ['9/10/2019', 24], ['9/11/2019', 27], ['9/12/2019', 27], ['9/13/2019', 37], ['9/14/2019', 20], ['9/15/2019', 37], ['9/16/2019', 23], ['9/17/2019', 32], ['9/18/2019', 29], ['9/19/2019', 29], ['9/20/2019', 28], ['9/21/2019', 35], ['9/22/2019', 38], ['9/23/2019', 37], ['9/24/2019', 32], ['9/25/2019', 33], ['9/26/2019', 34], ['9/27/2019', 31], ['9/28/2019', 37], ['9/29/2019', 35], ['9/30/2019', 20], ['10/1/2019', 19], ['10/2/2019', 24], ['10/3/2019', 27], ['10/4/2019', 20], ['10/5/2019', 33], ['10/6/2019', 38], ['10/7/2019', 28], ['10/8/2019', 25], ['10/9/2019', 19], ['10/10/2019', 26], ['10/11/2019', 33], ['10/12/2019', 27], ['10/13/2019', 30], ['10/14/2019', 27], ['10/15/2019', 22], ['10/16/2019', 27], ['10/17/2019', 24], ['10/18/2019', 22], ['10/19/2019', 27], ['10/20/2019', 29], ['10/21/2019', 21], ['10/22/2019', 21], ['10/23/2019', 24], ['10/24/2019', 22], ['10/25/2019', 29], ['10/26/2019', 30], ['10/27/2019', 21], ['10/28/2019', 27], ['10/29/2019', 29], ['10/30/2019', 28], ['10/31/2019', 30], ['11/1/2019', 20], ['11/2/2019', 25], ['11/3/2019', 22], ['11/4/2019', 24], ['11/5/2019', 24], ['11/6/2019', 21], ['11/7/2019', 27], ['11/8/2019', 27], ['11/9/2019', 26], ['11/10/2019', 25], ['11/11/2019', 28], ['11/12/2019', 22], ['11/13/2019', 27], ['11/14/2019', 20], ['11/15/2019', 27], ['11/16/2019', 30], ['11/17/2019', 23], ['11/18/2019', 21], ['11/19/2019', 25], ['11/20/2019', 30], ['11/21/2019', 28], ['11/22/2019', 24], ['11/23/2019', 27], ['11/24/2019', 25], ['11/25/2019', 20], ['11/26/2019', 27], ['11/27/2019', 22], ['11/28/2019', 30], ['11/29/2019', 28], ['11/30/2019', 26], ['12/1/2019', 20], ['12/2/2019', 24], ['12/3/2019', 20], ['12/4/2019', 26], ['12/5/2019', 22], ['12/6/2019', 23], ['12/7/2019', 21], ['12/8/2019', 30], ['12/9/2019', 30], ['12/10/2019', 26], ['12/11/2019', 27], ['12/12/2019', 23], ['12/13/2019', 25], ['12/14/2019', 21], ['12/15/2019', 22], ['12/16/2019', 25], ['12/17/2019', 30], ['12/18/2019', 21], ['12/19/2019', 21], ['12/20/2019', 22], ['12/21/2019', 22], ['12/22/2019', 27], ['12/23/2019', 30], ['12/24/2019', 23], ['12/25/2019', 21], ['12/26/2019', 24], ['12/27/2019', 29], ['12/28/2019', 23], ['12/29/2019', 28], ['12/30/2019', 30], ['12/31/2019', 25], ['1/1/2020', 19], ['1/2/2020', 25], ['1/3/2020', 26], ['1/4/2020', 37], ['1/5/2020', 37], ['1/6/2020', 27], ['1/7/2020', 21], ['1/8/2020', 30], ['1/9/2020', 24], ['1/10/2020', 27], ['1/11/2020', 33], ['1/12/2020', 37], ['1/13/2020', 24], ['1/14/2020', 30], ['1/15/2020', 31], ['1/16/2020', 26], ['1/17/2020', 29], ['1/18/2020', 27], ['1/19/2020', 37], ['1/20/2020', 36], ['1/21/2020', 38], ['1/22/2020', 38], ['1/23/2020', 19], ['1/24/2020', 31], ['1/25/2020', 19], ['1/26/2020', 23], ['1/27/2020', 30], ['1/28/2020', 36], ['1/29/2020', 30], ['1/30/2020', 20], ['1/31/2020', 26], ['2/1/2020', 33], ['2/2/2020', 34], ['2/3/2020', 23], ['2/4/2020', 33], ['2/5/2020', 23], ['2/6/2020', 32], ['2/7/2020', 34], ['2/8/2020', 30], ['2/9/2020', 34], ['2/10/2020', 32], ['2/11/2020', 24], ['2/12/2020', 28], ['2/13/2020', 35], ['2/14/2020', 38], ['2/15/2020', 25], ['2/16/2020', 33], ['2/17/2020', 30], ['2/18/2020', 32], ['2/19/2020', 25], ['2/20/2020', 29], ['2/21/2020', 23], ['2/22/2020', 22], ['2/23/2020', 28], ['2/24/2020', 20], ['2/25/2020', 27], ['2/26/2020', 32], ['2/27/2020', 33], ['2/28/2020', 34], ['2/29/2020', 31], ['3/1/2020', 22], ['3/2/2020', 34], ['3/3/2020', 32], ['3/4/2020', 23], ['3/5/2020', 20], ['3/6/2020', 22], ['3/7/2020', 38], ['3/8/2020', 22], ['3/9/2020', 35], ['3/10/2020', 26], ['3/11/2020', 38], ['3/12/2020', 32], ['3/13/2020', 22], ['3/14/2020', 22], ['3/15/2020', 37], ['3/16/2020', 21], ['3/17/2020', 23], ['3/18/2020', 21], ['3/19/2020', 24], ['3/20/2020', 20], ['3/21/2020', 26], ['3/22/2020', 25], ['3/23/2020', 20], ['3/24/2020', 23], ['3/25/2020', 27], ['3/26/2020', 25], ['3/27/2020', 30], ['3/28/2020', 25], ['3/29/2020', 20], ['3/30/2020', 27], ['3/31/2020', 20], ['4/1/2020', 20], ['4/2/2020', 23], ['4/3/2020', 26], ['4/4/2020', 24], ['4/5/2020', 29], ['4/6/2020', 21], ['4/7/2020', 21], ['4/8/2020', 25], ['4/9/2020', 23], ['4/10/2020', 27], ['4/11/2020', 24], ['4/12/2020', 23], ['4/13/2020', 26], ['4/14/2020', 28], ['4/15/2020', 22], ['4/16/2020', 24], ['4/17/2020', 26], ['4/18/2020', 24], ['4/19/2020', 20], ['4/20/2020', 21], ['4/21/2020', 24], ['4/22/2020', 20], ['4/23/2020', 28], ['4/24/2020', 28], ['4/25/2020', 29], ['4/26/2020', 27], ['4/27/2020', 29], ['4/28/2020', 22], ['4/29/2020', 25], ['4/30/2020', 22], ['5/1/2020', 29], ['5/2/2020', 23], ['5/3/2020', 29], ['5/4/2020', 26], ['5/5/2020', 24], ['5/6/2020', 20], ['5/7/2020', 20], ['5/8/2020', 22], ['5/9/2020', 23], ['5/10/2020', 24], ['5/11/2020', 21], ['5/12/2020', 30], ['5/13/2020', 21], ['5/14/2020', 20], ['5/15/2020', 28], ['5/16/2020', 25], ['5/17/2020', 26], ['5/18/2020', 24], ['5/19/2020', 22], ['5/20/2020', 28], ['5/21/2020', 20], ['5/22/2020', 26], ['5/23/2020', 24], ['5/24/2020', 20], ['5/25/2020', 29], ['5/26/2020', 30], ['5/27/2020', 24], ['5/28/2020', 24], ['5/29/2020', 23], ['5/30/2020', 21], ['5/31/2020', 20], ['6/1/2020', 30], ['6/2/2020', 21], ['6/3/2020', 23], ['6/4/2020', 24], ['6/5/2020', 32], ['6/6/2020', 28], ['6/7/2020', 21], ['6/8/2020', 19], ['6/9/2020', 32], ['6/10/2020', 30], ['6/11/2020', 29], ['6/12/2020', 36], ['6/13/2020', 22], ['6/14/2020', 28], ['6/15/2020', 37], ['6/16/2020', 26], ['6/17/2020', 36], ['6/18/2020', 22], ['6/19/2020', 25], ['6/20/2020', 23], ['6/21/2020', 20], ['6/22/2020', 25], ['6/23/2020', 20], ['6/24/2020', 26], ['6/25/2020', 33], ['6/26/2020', 19], ['6/27/2020', 30], ['6/28/2020', 24], ['6/29/2020', 26], ['6/30/2020', 34], ['7/1/2020', 19], ['7/2/2020', 20], ['7/3/2020', 22], ['7/4/2020', 35], ['7/5/2020', 24], ['7/6/2020', 37], ['7/7/2020', 30], ['7/8/2020', 34], ['7/9/2020', 38], ['7/10/2020', 37], ['7/11/2020', 21], ['7/12/2020', 31], ['7/13/2020', 28], ['7/14/2020', 36], ['7/15/2020', 21], ['7/16/2020', 28], ['7/17/2020', 19], ['7/18/2020', 27], ['7/19/2020', 22], ['7/20/2020', 22], ['7/21/2020', 28], ['7/22/2020', 25], ['7/23/2020', 23], ['7/24/2020', 23], ['7/25/2020', 26], ['7/26/2020', 25], ['7/27/2020', 26], ['7/28/2020', 26], ['7/29/2020', 25], ['7/30/2020', 30], ['7/31/2020', 30], ['8/1/2020', 24], ['8/2/2020', 30], ['8/3/2020', 24], ['8/4/2020', 21], ['8/5/2020', 21], ['8/6/2020', 26], ['8/7/2020', 25], ['8/8/2020', 27], ['8/9/2020', 21], ['8/10/2020', 29], ['8/11/2020', 21], ['8/12/2020', 22], ['8/13/2020', 30], ['8/14/2020', 23], ['8/15/2020', 29], ['8/16/2020', 24], ['8/17/2020', 25], ['8/18/2020', 27], ['8/19/2020', 29], ['8/20/2020', 25], ['8/21/2020', 25], ['8/22/2020', 20], ['8/23/2020', 24], ['8/24/2020', 21], ['8/25/2020', 27], ['8/26/2020', 28], ['8/27/2020', 30], ['8/28/2020', 20], ['8/29/2020', 30], ['8/30/2020', 28], ['8/31/2020', 25], ['9/1/2020', 24], ['9/2/2020', 23], ['9/3/2020', 22], ['9/4/2020', 27], ['9/5/2020', 25], ['9/6/2020', 22], ['9/7/2020', 30], ['9/8/2020', 26], ['9/9/2020', 27], ['9/10/2020', 23], ['9/11/2020', 23], ['9/12/2020', 26], ['9/13/2020', 26], ['9/14/2020', 24], ['9/15/2020', 21], ['9/16/2020', 20], ['9/17/2020', 25], ['9/18/2020', 27], ['9/19/2020', 30], ['9/20/2020', 22], ['9/21/2020', 27], ['9/22/2020', 24], ['9/23/2020', 21], ['9/24/2020', 26], ['9/25/2020', 25], ['9/26/2020', 29], ['9/27/2020', 24], ['9/28/2020', 22], ['9/29/2020', 25], ['9/30/2020', 21], ['10/1/2020', 23], ['10/2/2020', 24], ['10/3/2020', 24], ['10/4/2020', 20], ['10/5/2020', 23], ['10/6/2020', 37], ['10/7/2020', 22], ['10/8/2020', 30], ['10/9/2020', 25], ['10/10/2020', 35], ['10/11/2020', 34], ['10/12/2020', 25], ['10/13/2020', 23], ['10/14/2020', 37], ['10/15/2020', 36], ['10/16/2020', 21], ['10/17/2020', 24], ['10/18/2020', 37], ['10/19/2020', 20], ['10/20/2020', 34], ['10/21/2020', 32], ['10/22/2020', 29], ['10/23/2020', 36], ['10/24/2020', 38], ['10/25/2020', 19], ['10/26/2020', 37], ['10/27/2020', 32], ['10/28/2020', 27], ['10/29/2020', 34], ['10/30/2020', 34], ['10/31/2020', 35], ['11/1/2020', 24], ['11/2/2020', 34], ['11/3/2020', 25], ['11/4/2020', 24], ['11/5/2020', 23], ['11/6/2020', 27], ['11/7/2020', 32], ['11/8/2020', 26], ['11/9/2020', 23], ['11/10/2020', 29], ['11/11/2020', 24], ['11/12/2020', 31], ['11/13/2020', 34], ['11/14/2020', 24], ['11/15/2020', 20], ['11/16/2020', 37], ['11/17/2020', 23], ['11/18/2020', 30], ['11/19/2020', 21], ['11/20/2020', 26], ['11/21/2020', 24], ['11/22/2020', 27], ['11/23/2020', 24], ['11/24/2020', 22], ['11/25/2020', 22], ['11/26/2020', 26], ['11/27/2020', 29], ['11/28/2020', 24], ['11/29/2020', 26], ['11/30/2020', 24], ['12/1/2020', 26], ['12/2/2020', 24], ['12/3/2020', 22], ['12/4/2020', 29], ['12/5/2020', 22], ['12/6/2020', 30], ['12/7/2020', 21], ['12/8/2020', 28], ['12/9/2020', 24], ['12/10/2020', 23], ['12/11/2020', 29], ['12/12/2020', 25], ['12/13/2020', 27], ['12/14/2020', 21], ['12/15/2020', 26], ['12/16/2020', 24], ['12/17/2020', 30], ['12/18/2020', 20], ['12/19/2020', 29], ['12/20/2020', 21], ['12/21/2020', 25], ['12/22/2020', 22], ['12/23/2020', 22], ['12/24/2020', 30], ['12/25/2020', 20], ['12/26/2020', 23], ['12/27/2020', 27], ['12/28/2020', 22], ['12/29/2020', 24], ['12/30/2020', 28], ['12/31/2020', 20], ['1/1/2021', 27], ['1/2/2021', 26], ['1/3/2021', 26], ['1/4/2021', 20], ['1/5/2021', 21], ['1/6/2021', 23], ['1/7/2021', 26], ['1/8/2021', 26], ['1/9/2021', 23], ['1/10/2021', 20], ['1/11/2021', 21], ['1/12/2021', 27], ['1/13/2021', 27], ['1/14/2021', 21], ['1/15/2021', 30], ['1/16/2021', 24], ['1/17/2021', 27], ['1/18/2021', 26], ['1/19/2021', 25], ['1/20/2021', 28], ['1/21/2021', 30], ['1/22/2021', 20], ['1/23/2021', 27], ['1/24/2021', 21], ['1/25/2021', 22], ['1/26/2021', 29], ['1/27/2021', 22], ['1/28/2021', 25], ['1/29/2021', 21], ['1/30/2021', 20], ['1/31/2021', 30], ['2/1/2021', 28], ['2/2/2021', 21], ['2/3/2021', 22], ['2/4/2021', 25], ['2/5/2021', 23], ['2/6/2021', 20], ['2/7/2021', 37], ['2/8/2021', 30], ['2/9/2021', 28], ['2/10/2021', 27], ['2/11/2021', 37], ['2/12/2021', 21], ['2/13/2021', 35], ['2/14/2021', 33], ['2/15/2021', 38], ['2/16/2021', 35], ['2/17/2021', 30], ['2/18/2021', 37], ['2/19/2021', 35], ['2/20/2021', 31], ['2/21/2021', 35], ['2/22/2021', 29], ['2/23/2021', 35], ['2/24/2021', 29], ['2/25/2021', 27], ['2/26/2021', 21], ['2/27/2021', 26], ['2/28/2021', 23], ['3/1/2021', 26], ['3/2/2021', 20], ['3/3/2021', 19], ['3/4/2021', 25], ['3/5/2021', 33], ['3/6/2021', 33], ['3/7/2021', 32], ['3/8/2021', 24], ['3/9/2021', 25], ['3/10/2021', 19], ['3/11/2021', 36], ['3/12/2021', 30], ['3/13/2021', 26], ['3/14/2021', 25], ['3/15/2021', 25], ['3/16/2021', 24], ['3/17/2021', 34], ['3/18/2021', 35], ['3/19/2021', 34], ['3/20/2021', 38], ['3/21/2021', 25], ['3/22/2021', 29], ['3/23/2021', 20], ['3/24/2021', 20], ['3/25/2021', 23], ['3/26/2021', 27], ['3/27/2021', 27], ['3/28/2021', 22], ['3/29/2021', 24], ['3/30/2021', 25], ['3/31/2021', 26], ['4/1/2021', 24], ['4/2/2021', 21], ['4/3/2021', 25], ['4/4/2021', 26], ['4/5/2021', 23], ['4/6/2021', 30], ['4/7/2021', 22], ['4/8/2021', 29], ['4/9/2021', 22], ['4/10/2021', 23], ['4/11/2021', 25], ['4/12/2021', 28], ['4/13/2021', 25], ['4/14/2021', 27], ['4/15/2021', 20], ['4/16/2021', 30], ['4/17/2021', 29], ['4/18/2021', 25], ['4/19/2021', 30], ['4/20/2021', 24], ['4/21/2021', 30], ['4/22/2021', 30], ['4/23/2021', 20], ['4/24/2021', 20], ['4/25/2021', 28], ['4/26/2021', 23], ['4/27/2021', 25], ['4/28/2021', 22], ['4/29/2021', 28], ['4/30/2021', 28], ['5/1/2021', 27], ['5/2/2021', 28], ['5/3/2021', 22], ['5/4/2021', 23], ['5/5/2021', 23], ['5/6/2021', 29], ['5/7/2021', 28], ['5/8/2021', 30], ['5/9/2021', 24], ['5/10/2021', 24], ['5/11/2021', 27], ['5/12/2021', 20], ['5/13/2021', 29], ['5/14/2021', 23], ['5/15/2021', 24], ['5/16/2021', 30], ['5/17/2021', 20], ['5/18/2021', 22], ['5/19/2021', 29], ['5/20/2021', 21], ['5/21/2021', 21], ['5/22/2021', 29], ['5/23/2021', 24], ['5/24/2021', 23], ['5/25/2021', 28], ['5/26/2021', 22], ['5/27/2021', 30], ['5/28/2021', 26], ['5/29/2021', 22], ['5/30/2021', 28], ['5/31/2021', 21], ['6/1/2021', 27], ['6/2/2021', 28], ['6/3/2021', 29], ['6/4/2021', 28], ['6/5/2021', 30], ['6/6/2021', 21], ['6/7/2021', 22], ['6/8/2021', 34], ['6/9/2021', 37], ['6/10/2021', 35], ['6/11/2021', 29], ['6/12/2021', 21], ['6/13/2021', 34], ['6/14/2021', 28], ['6/15/2021', 30], ['6/16/2021', 34], ['6/17/2021', 32], ['6/18/2021', 24], ['6/19/2021', 33], ['6/20/2021', 27], ['6/21/2021', 21], ['6/22/2021', 19], ['6/23/2021', 25], ['6/24/2021', 25], ['6/25/2021', 31], ['6/26/2021', 33], ['6/27/2021', 19], ['6/28/2021', 38], ['6/29/2021', 36], ['6/30/2021', 27], ['7/1/2021', 33], ['7/2/2021', 38], ['7/3/2021', 26], ['7/4/2021', 22], ['7/5/2021', 35], ['7/6/2021', 23], ['7/7/2021', 38], ['7/8/2021', 31], ['7/9/2021', 21], ['7/10/2021', 31], ['7/11/2021', 32], ['7/12/2021', 21], ['7/13/2021', 32], ['7/14/2021', 23], ['7/15/2021', 33], ['7/16/2021', 22], ['7/17/2021', 28], ['7/18/2021', 23], ['7/19/2021', 30], ['7/20/2021', 36], ['7/21/2021', 28], ['7/22/2021', 20], ['7/23/2021', 25], ['7/24/2021', 28], ['7/25/2021', 25], ['7/26/2021', 27], ['7/27/2021', 24], ['7/28/2021', 20], ['7/29/2021', 26], ['7/30/2021', 28], ['7/31/2021', 27], ['8/1/2021', 22], ['8/2/2021', 30], ['8/3/2021', 24], ['8/4/2021', 22], ['8/5/2021', 23], ['8/6/2021', 24], ['8/7/2021', 28], ['8/8/2021', 20], ['8/9/2021', 24], ['8/10/2021', 22], ['8/11/2021', 28], ['8/12/2021', 22], ['8/13/2021', 27], ['8/14/2021', 21], ['8/15/2021', 24], ['8/16/2021', 27], ['8/17/2021', 25], ['8/18/2021', 26], ['8/19/2021', 23], ['8/20/2021', 24], ['8/21/2021', 24], ['8/22/2021', 22], ['8/23/2021', 24], ['8/24/2021', 26], ['8/25/2021', 29], ['8/26/2021', 29], ['8/27/2021', 26], ['8/28/2021', 21], ['8/29/2021', 25], ['8/30/2021', 27], ['8/31/2021', 22], ['9/1/2021', 23], ['9/2/2021', 26], ['9/3/2021', 25], ['9/4/2021', 28], ['9/5/2021', 29], ['9/6/2021', 22], ['9/7/2021', 27], ['9/8/2021', 27], ['9/9/2021', 29], ['9/10/2021', 29], ['9/11/2021', 25], ['9/12/2021', 20], ['9/13/2021', 22], ['9/14/2021', 26], ['9/15/2021', 24], ['9/16/2021', 27], ['9/17/2021', 27], ['9/18/2021', 30], ['9/19/2021', 21], ['9/20/2021', 30], ['9/21/2021', 27], ['9/22/2021', 30], ['9/23/2021', 21], ['9/24/2021', 22], ['9/25/2021', 30], ['9/26/2021', 30], ['9/27/2021', 21], ['9/28/2021', 30], ['9/29/2021', 20], ['9/30/2021', 25], ['10/1/2021', 26], ['10/2/2021', 23], ['10/3/2021', 20], ['10/4/2021', 21], ['10/5/2021', 24], ['10/6/2021', 30], ['10/7/2021', 27], ['10/8/2021', 25], ['10/9/2021', 27], ['10/10/2021', 33], ['10/11/2021', 24], ['10/12/2021', 35], ['10/13/2021', 26], ['10/14/2021', 24], ['10/15/2021', 24], ['10/16/2021', 38], ['10/17/2021', 23], ['10/18/2021', 29], ['10/19/2021', 32], ['10/20/2021', 34], ['10/21/2021', 31], ['10/22/2021', 25], ['10/23/2021', 22], ['10/24/2021', 27], ['10/25/2021', 30], ['10/26/2021', 24], ['10/27/2021', 33], ['10/28/2021', 35], ['10/29/2021', 28], ['10/30/2021', 27], ['10/31/2021', 38], ['11/1/2021', 24], ['11/2/2021', 33], ['11/3/2021', 36], ['11/4/2021', 34], ['11/5/2021', 34], ['11/6/2021', 24], ['11/7/2021', 31], ['11/8/2021', 38], ['11/9/2021', 29], ['11/10/2021', 38], ['11/11/2021', 27], ['11/12/2021', 19], ['11/13/2021', 26], ['11/14/2021', 33], ['11/15/2021', 20], ['11/16/2021', 35], ['11/17/2021', 37], ['11/18/2021', 28], ['11/19/2021', 36], ['11/20/2021', 37], ['11/21/2021', 26], ['11/22/2021', 26], ['11/23/2021', 28], ['11/24/2021', 28], ['11/25/2021', 30], ['11/26/2021', 26], ['11/27/2021', 28], ['11/28/2021', 27], ['11/29/2021', 33], ['11/30/2021', 28], ['12/1/2021', 27], ['12/2/2021', 24], ['12/3/2021', 23], ['12/4/2021', 28], ['12/5/2021', 36], ['12/6/2021', 32], ['12/7/2021', 23], ['12/8/2021', 25], ['12/9/2021', 38], ['12/10/2021', 22], ['12/11/2021', 33], ['12/12/2021', 30], ['12/13/2021', 26], ['12/14/2021', 33], ['12/15/2021', 22], ['12/16/2021', 24], ['12/17/2021', 23], ['12/18/2021', 37], ['12/19/2021', 24], ['12/20/2021', 26], ['12/21/2021', 36], ['12/22/2021', 32], ['12/23/2021', 31], ['12/24/2021', 22], ['12/25/2021', 35], ['12/26/2021', 24], ['12/27/2021', 23], ['12/28/2021', 23], ['12/29/2021', 21], ['12/30/2021', 28], ['12/31/2021', 22], ['1/1/2022', 26], ['1/2/2022', 30], ['1/3/2022', 21], ['1/4/2022', 25], ['1/5/2022', 25], ['1/6/2022', 22], ['1/7/2022', 20], ['1/8/2022', 28], ['1/9/2022', 23], ['1/10/2022', 29], ['1/11/2022', 20], ['1/12/2022', 24], ['1/13/2022', 22], ['1/14/2022', 28], ['1/15/2022', 22], ['1/16/2022', 25], ['1/17/2022', 26], ['1/18/2022', 21], ['1/19/2022', 20], ['1/20/2022', 30], ['1/21/2022', 27], ['1/22/2022', 21], ['1/23/2022', 26], ['1/24/2022', 28], ['1/25/2022', 29], ['1/26/2022', 30], ['1/27/2022', 24], ['1/28/2022', 20], ['1/29/2022', 22], ['1/30/2022', 30], ['1/31/2022', 30], ['2/1/2022', 21], ['2/2/2022', 29], ['2/3/2022', 22], ['2/4/2022', 27], ['2/5/2022', 26], ['2/6/2022', 30], ['2/7/2022', 29], ['2/8/2022', 27], ['2/9/2022', 26], ['2/10/2022', 22], ['2/11/2022', 25], ['2/12/2022', 23], ['2/13/2022', 24], ['2/14/2022', 21], ['2/15/2022', 23], ['2/16/2022', 21], ['2/17/2022', 20], ['2/18/2022', 20], ['2/19/2022', 29], ['2/20/2022', 20], ['2/21/2022', 22], ['2/22/2022', 26], ['2/23/2022', 26], ['2/24/2022', 29], ['2/25/2022', 30], ['2/26/2022', 26], ['2/27/2022', 22], ['2/28/2022', 29], ['3/1/2022', 20], ['3/2/2022', 26], ['3/3/2022', 25], ['3/4/2022', 22], ['3/5/2022', 27], ['3/6/2022', 27], ['3/7/2022', 24], ['3/8/2022', 20], ['3/9/2022', 27], ['3/10/2022', 27], ['3/11/2022', 30], ['3/12/2022', 24], ['3/13/2022', 28], ['3/14/2022', 22], ['3/15/2022', 33], ['3/16/2022', 26], ['3/17/2022', 35], ['3/18/2022', 34], ['3/19/2022', 35], ['3/20/2022', 32], ['3/21/2022', 32], ['3/22/2022', 28], ['3/23/2022', 35], ['3/24/2022', 20], ['3/25/2022', 30], ['3/26/2022', 31], ['3/27/2022', 21], ['3/28/2022', 32], ['3/29/2022', 32], ['3/30/2022', 20], ['3/31/2022', 27], ['4/1/2022', 35], ['4/2/2022', 27], ['4/3/2022', 23], ['4/4/2022', 19], ['4/5/2022', 24], ['4/6/2022', 24], ['4/7/2022', 36], ['4/8/2022', 28], ['4/9/2022', 22], ['4/10/2022', 27], ['4/11/2022', 34], ['4/12/2022', 36], ['4/13/2022', 21], ['4/14/2022', 36], ['4/15/2022', 34], ['4/16/2022', 21], ['4/17/2022', 21], ['4/18/2022', 27], ['4/19/2022', 29], ['4/20/2022', 25], ['4/21/2022', 22], ['4/22/2022', 34], ['4/23/2022', 33], ['4/24/2022', 23], ['4/25/2022', 20], ['4/26/2022', 33], ['4/27/2022', 21], ['4/28/2022', 22], ['4/29/2022', 23], ['4/30/2022', 28], ['5/1/2022', 21], ['5/2/2022', 25], ['5/3/2022', 28], ['5/4/2022', 27], ['5/5/2022', 24], ['5/6/2022', 29], ['5/7/2022', 22], ['5/8/2022', 25], ['5/9/2022', 28], ['5/10/2022', 27], ['5/11/2022', 21], ['5/12/2022', 25], ['5/13/2022', 26], ['5/14/2022', 24], ['5/15/2022', 25], ['5/16/2022', 28], ['5/17/2022', 24], ['5/18/2022', 23], ['5/19/2022', 25], ['5/20/2022', 27], ['5/21/2022', 22], ['5/22/2022', 25], ['5/23/2022', 26], ['5/24/2022', 24], ['5/25/2022', 23], ['5/26/2022', 20], ['5/27/2022', 25], ['5/28/2022', 24], ['5/29/2022', 29], ['5/30/2022', 22], ['5/31/2022', 28], ['6/1/2022', 26], ['6/2/2022', 29], ['6/3/2022', 21], ['6/4/2022', 26], ['6/5/2022', 26], ['6/6/2022', 24], ['6/7/2022', 29], ['6/8/2022', 28], ['6/9/2022', 20], ['6/10/2022', 20], ['6/11/2022', 21], ['6/12/2022', 21], ['6/13/2022', 25], ['6/14/2022', 25], ['6/15/2022', 22], ['6/16/2022', 22], ['6/17/2022', 28], ['6/18/2022', 20], ['6/19/2022', 25], ['6/20/2022', 30], ['6/21/2022', 24], ['6/22/2022', 23], ['6/23/2022', 28], ['6/24/2022', 21], ['6/25/2022', 22], ['6/26/2022', 27], ['6/27/2022', 29], ['6/28/2022', 27], ['6/29/2022', 25], ['6/30/2022', 25], ['7/1/2022', 23], ['7/2/2022', 30], ['7/3/2022', 23], ['7/4/2022', 21], ['7/5/2022', 21], ['7/6/2022', 23], ['7/7/2022', 23], ['7/8/2022', 25], ['7/9/2022', 23], ['7/10/2022', 21], ['7/11/2022', 22], ['7/12/2022', 23], ['7/13/2022', 22], ['7/14/2022', 26], ['7/15/2022', 29], ['7/16/2022', 22], ['7/17/2022', 36], ['7/18/2022', 37], ['7/19/2022', 32], ['7/20/2022', 26], ['7/21/2022', 28], ['7/22/2022', 25], ['7/23/2022', 29], ['7/24/2022', 32], ['7/25/2022', 32], ['7/26/2022', 36], ['7/27/2022', 23], ['7/28/2022', 30], ['7/29/2022', 29], ['7/30/2022', 37], ['7/31/2022', 26], ['8/1/2022', 26], ['8/2/2022', 33], ['8/3/2022', 28], ['8/4/2022', 32], ['8/5/2022', 36], ['8/6/2022', 24], ['8/7/2022', 30], ['8/8/2022', 34], ['8/9/2022', 33], ['8/10/2022', 23], ['8/11/2022', 37], ['8/12/2022', 38], ['8/13/2022', 27], ['8/14/2022', 28], ['8/15/2022', 36], ['8/16/2022', 36], ['8/17/2022', 28], ['8/18/2022', 33], ['8/19/2022', 33], ['8/20/2022', 27], ['8/21/2022', 38], ['8/22/2022', 37], ['8/23/2022', 34], ['8/24/2022', 30], ['8/25/2022', 25], ['8/26/2022', 31], ['8/27/2022', 30], ['8/28/2022', 30], ['8/29/2022', 20], ['8/30/2022', 27], ['8/31/2022', 26], ['9/1/2022', 27], ['9/2/2022', 26], ['9/3/2022', 22], ['9/4/2022', 25], ['9/5/2022', 20], ['9/6/2022', 23], ['9/7/2022', 20], ['9/8/2022', 30], ['9/9/2022', 23], ['9/10/2022', 25], ['9/11/2022', 20], ['9/12/2022', 26], ['9/13/2022', 21], ['9/14/2022', 27], ['9/15/2022', 23], ['9/16/2022', 23], ['9/17/2022', 30], ['9/18/2022', 27], ['9/19/2022', 20], ['9/20/2022', 24], ['9/21/2022', 28], ['9/22/2022', 29], ['9/23/2022', 25], ['9/24/2022', 23], ['9/25/2022', 29], ['9/26/2022', 23], ['9/27/2022', 24], ['9/28/2022', 22], ['9/29/2022', 23], ['9/30/2022', 22], ['10/1/2022', 21], ['10/2/2022', 24], ['10/3/2022', 23], ['10/4/2022', 30], ['10/5/2022', 30], ['10/6/2022', 25], ['10/7/2022', 23], ['10/8/2022', 25], ['10/9/2022', 27], ['10/10/2022', 24], ['10/11/2022', 21], ['10/12/2022', 22], ['10/13/2022', 30], ['10/14/2022', 28], ['10/15/2022', 30], ['10/16/2022', 20], ['10/17/2022', 29], ['10/18/2022', 24], ['10/19/2022', 29], ['10/20/2022', 23], ['10/21/2022', 27], ['10/22/2022', 21], ['10/23/2022', 28], ['10/24/2022', 24], ['10/25/2022', 26], ['10/26/2022', 23], ['10/27/2022', 20], ['10/28/2022', 27], ['10/29/2022', 27], ['10/30/2022', 29], ['10/31/2022', 22], ['11/1/2022', 29], ['11/2/2022', 28], ['11/3/2022', 29], ['11/4/2022', 29], ['11/5/2022', 30], ['11/6/2022', 20], ['11/7/2022', 21], ['11/8/2022', 26], ['11/9/2022', 30], ['11/10/2022', 23], ['11/11/2022', 25], ['11/12/2022', 27], ['11/13/2022', 29], ['11/14/2022', 29], ['11/15/2022', 26], ['11/16/2022', 27], ['11/17/2022', 26], ['11/18/2022', 27], ['11/19/2022', 21], ['11/20/2022', 37], ['11/21/2022', 23], ['11/22/2022', 27], ['11/23/2022', 21], ['11/24/2022', 36], ['11/25/2022', 29], ['11/26/2022', 26], ['11/27/2022', 25], ['11/28/2022', 38], ['11/29/2022', 28], ['11/30/2022', 22], ['12/1/2022', 33], ['12/2/2022', 27], ['12/3/2022', 20], ['12/4/2022', 21], ['12/5/2022', 21], ['12/6/2022', 36], ['12/7/2022', 29], ['12/8/2022', 23], ['12/9/2022', 27], ['12/10/2022', 21], ['12/11/2022', 38], ['12/12/2022', 35], ['12/13/2022', 23], ['12/14/2022', 20], ['12/15/2022', 30], ['12/16/2022', 20], ['12/17/2022', 23], ['12/18/2022', 27], ['12/19/2022', 19], ['12/20/2022', 33], ['12/21/2022', 30], ['12/22/2022', 23], ['12/23/2022', 38], ['12/24/2022', 24], ['12/25/2022', 29], ['12/26/2022', 38], ['12/27/2022', 29], ['12/28/2022', 27], ['12/29/2022', 27], ['12/30/2022', 27], ['12/31/2022', 27], ['1/1/2023', 24], ['1/2/2023', 26], ['1/3/2023', 21], ['1/4/2023', 26], ['1/5/2023', 23], ['1/6/2023', 28], ['1/7/2023', 26], ['1/8/2023', 29], ['1/9/2023', 27], ['1/10/2023', 24], ['1/11/2023', 28], ['1/12/2023', 20], ['1/13/2023', 29], ['1/14/2023', 26], ['1/15/2023', 29], ['1/16/2023', 23], ['1/17/2023', 25], ['1/18/2023', 27], ['1/19/2023', 26], ['1/20/2023', 21], ['1/21/2023', 26], ['1/22/2023', 21], ['1/23/2023', 22], ['1/24/2023', 26], ['1/25/2023', 30], ['1/26/2023', 27], ['1/27/2023', 23], ['1/28/2023', 22], ['1/29/2023', 27], ['1/30/2023', 26], ['1/31/2023', 21], ['2/1/2023', 25], ['2/2/2023', 25], ['2/3/2023', 20], ['2/4/2023', 21], ['2/5/2023', 23], ['2/6/2023', 26], ['2/7/2023', 22], ['2/8/2023', 23], ['2/9/2023', 28], ['2/10/2023', 23], ['2/11/2023', 27], ['2/12/2023', 26], ['2/13/2023', 21], ['2/14/2023', 26], ['2/15/2023', 22], ['2/16/2023', 23], ['2/17/2023', 20], ['2/18/2023', 23], ['2/19/2023', 25], ['2/20/2023', 28], ['2/21/2023', 23], ['2/22/2023', 22], ['2/23/2023', 20], ['2/24/2023', 22], ['2/25/2023', 29], ['2/26/2023', 20], ['2/27/2023', 25], ['2/28/2023', 28], ['3/1/2023', 24], ['3/2/2023', 21], ['3/3/2023', 28], ['3/4/2023', 23], ['3/5/2023', 28], ['3/6/2023', 22], ['3/7/2023', 27], ['3/8/2023', 26], ['3/9/2023', 27], ['3/10/2023', 26], ['3/11/2023', 22], ['3/12/2023', 22], ['3/13/2023', 29], ['3/14/2023', 24], ['3/15/2023', 27], ['3/16/2023', 29], ['3/17/2023', 28], ['3/18/2023', 28], ['3/19/2023', 37], ['3/20/2023', 35], ['3/21/2023', 25], ['3/22/2023', 21], ['3/23/2023', 20], ['3/24/2023', 23], ['3/25/2023', 24], ['3/26/2023', 32], ['3/27/2023', 24], ['3/28/2023', 22], ['3/29/2023', 28], ['3/30/2023', 33], ['3/31/2023', 24], ['4/1/2023', 23], ['4/2/2023', 27], ['4/3/2023', 24], ['4/4/2023', 21], ['4/5/2023', 33], ['4/6/2023', 19], ['4/7/2023', 23], ['4/8/2023', 28], ['4/9/2023', 37], ['4/10/2023', 23], ['4/11/2023', 38], ['4/12/2023', 25], ['4/13/2023', 33], ['4/14/2023', 33], ['4/15/2023', 37], ['4/16/2023', 21], ['4/17/2023', 29], ['4/18/2023', 38], ['4/19/2023', 28], ['4/20/2023', 20], ['4/21/2023', 27], ['4/22/2023', 33], ['4/23/2023', 19], ['4/24/2023', 26], ['4/25/2023', 30], ['4/26/2023', 34], ['4/27/2023', 32], ['4/28/2023', 21], ['4/29/2023', 33], ['4/30/2023', 37], ['5/1/2023', 22], ['5/2/2023', 27], ['5/3/2023', 21], ['5/4/2023', 30], ['5/5/2023', 30], ['5/6/2023', 28], ['5/7/2023', 23], ['5/8/2023', 30], ['5/9/2023', 26], ['5/10/2023', 30], ['5/11/2023', 27], ['5/12/2023', 22], ['5/13/2023', 24], ['5/14/2023', 25], ['5/15/2023', 30], ['5/16/2023', 25], ['5/17/2023', 26], ['5/18/2023', 22], ['5/19/2023', 26], ['5/20/2023', 26], ['5/21/2023', 23], ['5/22/2023', 29], ['5/23/2023', 23], ['5/24/2023', 28], ['5/25/2023', 24], ['5/26/2023', 26], ['5/27/2023', 27], ['5/28/2023', 21], ['5/29/2023', 23], ['5/30/2023', 30], ['5/31/2023', 28], ['6/1/2023', 20], ['6/2/2023', 20], ['6/3/2023', 27], ['6/4/2023', 29], ['6/5/2023', 21], ['6/6/2023', 23], ['6/7/2023', 23], ['6/8/2023', 30], ['6/9/2023', 26], ['6/10/2023', 20], ['6/11/2023', 21], ['6/12/2023', 21], ['6/13/2023', 28], ['6/14/2023', 21], ['6/15/2023', 27], ['6/16/2023', 26], ['6/17/2023', 22], ['6/18/2023', 26], ['6/19/2023', 23], ['6/20/2023', 20], ['6/21/2023', 24], ['6/22/2023', 29], ['6/23/2023', 27], ['6/24/2023', 23], ['6/25/2023', 21], ['6/26/2023', 25], ['6/27/2023', 30], ['6/28/2023', 26], ['6/29/2023', 23], ['6/30/2023', 23], ['7/1/2023', 21], ['7/2/2023', 22], ['7/3/2023', 30], ['7/4/2023', 24], ['7/5/2023', 25], ['7/6/2023', 20], ['7/7/2023', 26], ['7/8/2023', 21], ['7/9/2023', 26], ['7/10/2023', 26], ['7/11/2023', 24], ['7/12/2023', 21], ['7/13/2023', 25], ['7/14/2023', 23], ['7/15/2023', 25], ['7/16/2023', 30], ['7/17/2023', 28], ['7/18/2023', 28], ['7/19/2023', 26], ['7/20/2023', 21], ['7/21/2023', 36], ['7/22/2023', 27], ['7/23/2023', 26], ['7/24/2023', 25], ['7/25/2023', 30], ['7/26/2023', 19], ['7/27/2023', 26], ['7/28/2023', 19], ['7/29/2023', 31], ['7/30/2023', 35], ['7/31/2023', 20], ['8/1/2023', 31], ['8/2/2023', 20]]


dfh = pd.DataFrame(mylist)

from fbprophet import Prophet
# instantiate the model and set parameters

model = Prophet(

    interval_width=0.95,

    growth='linear',

    daily_seasonality=False,

    weekly_seasonality=True,

    yearly_seasonality=True,

    seasonality_mode='multiplicative'

)

 

# fit the model to historical data

model.fit(dfh)

 

 

future_pd = model.make_future_dataframe(

    periods=90,

    freq='d',

    include_history=True

)

 

# predict over the dataset

forecast_pd = model.predict(future_pd)


predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='sales')

display(fig)
