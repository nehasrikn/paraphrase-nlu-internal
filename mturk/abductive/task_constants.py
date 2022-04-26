TAB_INSTRUCTIONS="""
<b> Paraphrase the sentence above in two different ways</b> such that its meaning is retained, but the words themselves substantially differ from those in the original sentence.
"""

INPUT_TEMPLATE="""
<crowd-input id='paraphrase_ID' name='paraphrase_ID' spellcheck="true" label='Paraphrase #NUM' required onchange onpropertychange onkeyuponpaste oninput="calculate_BLEU('free_text_id', 'similarity_id', 'original_sentence')"></crowd-input>
<div>Distance from Original Sentence: <span id='similarity_ID'></span></div>
"""


TABS = """
<div id=task1 class='first-container'>
			<div class="left-container">
				<h3>Beginning</h3>
			    <crowd-card>
			      <div class="card">
			        OBSERVATION_1
			      </div>
			    </crowd-card>
			</div>

			<div class="middle-container">
				<h3>Middle</h3>
				<crowd-tabs>
					<crowd-tab header="Sentence 1 (Plausible Middle)">
						<crowd-card>
					      <div class="card">
					        HYPOTHESIS_CORRECT
					      </div>
					    </crowd-card>
						<div style='text-align: left; width: 600px; padding-left: 5px; padding-top: 30px'>
							TAB_INSTRUCTIONS
							TABS_CORRECT
							
						</div>
					</crowd-tab>

					<crowd-tab header='Sentence 2 (Implausible Middle)'>
						<crowd-card>
					      <div class="card">
					        HYPOTHESIS_INCORRECT
					      </div>
				   		</crowd-card>
						<div style='text-align: left; width: 600px; padding-left: 5px; padding-top: 30px'>
							TAB_INSTRUCTIONS
							TABS_INCORRECT
						</div>

					</crowd-tab>


					</crowd-tab>
				</crowd-tabs>
			    

			</div>

			<!-- <span id="vertical-separator"></span> -->
			<div class="right-container">
				<h3>Ending</h3>
				<crowd-card>
			      <div class="card">
			        OBSERVATION_2
			      </div>
			    </crowd-card>
			</div>
	</div>
	<br>
	<div style="position: relative;">
		<crowd-button id="submit_button" form-action="submit">
			Submit
		</crowd-button>
		<div id="errorBox"></div>
		<br>
	</div>

"""